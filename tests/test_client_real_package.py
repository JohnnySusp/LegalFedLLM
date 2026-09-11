from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import httpx

from client.main import CoordinatorGateway, create_app as create_client_app
from client.model_profiles import QWEN_PROFILE_ID, pinned_client_profile
from client.runtime import ClientRuntime, ClientRuntimeError
from client.training import (
    LocalTrainingRecord,
    TrainingExecutionProfile,
    load_private_examples,
    private_dataset_semantic_hash,
)
from host.model_profiles import pinned_host_profile
from shared.knowledge_artifact import load_package_samples
from shared.prompt import PROMPT_TEMPLATE
from shared.protocol import KnowledgePackage, KnowledgeSample, RoundManifest, utc_text
from shared.reference_dataset import (
    ReferenceSample,
    reference_dataset_identity,
    write_reference_jsonl,
)
from tests.test_round import Stack


class RealPackagePipelineTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _write_adapter(staging: Path, marker: bytes) -> None:
        (staging / "adapter_config.json").write_text(
            json.dumps({"format": "test-peft"}),
            encoding="utf-8",
        )
        (staging / "adapter_model.safetensors").write_bytes(marker)

    @staticmethod
    def _knowledge(manifest: RoundManifest) -> list[KnowledgeSample]:
        samples: list[KnowledgeSample] = []
        for index, sample_id in enumerate(manifest.sample_ids):
            source = [100 + index, 200 + index, 300 + index]
            samples.append(
                KnowledgeSample(
                    sample_id=sample_id,
                    source_input_ids=source,
                    attention_length=len(source),
                    top_k_token_ids=[[10 + index, 20 + index] for _ in source],
                    top_k_logits=[[2.5 + index, 1.5 + index] for _ in source],
                    full_logsumexp=[
                        3.75 + (2 * index), 5.5 + index, 5.5 + index
                    ],
                    gold_token_ids=[10 + index, -100, -100],
                    gold_token_logits=[2.5 + index, 0.0, 0.0],
                    gold_token_nll=[1.25 + index, 0.0, 0.0],
                    ce_loss=1.25 + index,
                )
            )
        return samples

    async def _fixture(self, root: Path):
        private_marker = "PRIVATE-CLIENT-A-MARKER-DO-NOT-EXPORT"
        private_answer = "PRIVATE-CLIENT-A-ANSWER-DO-NOT-EXPORT"
        private_path = root / "private.jsonl"
        private_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "example_id": "private-a-1",
                    "prompt": private_marker,
                    "answer": private_answer,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        reference_samples = [
            ReferenceSample(
                schema_version=1,
                dataset_id="client-real-reference",
                dataset_version="v1",
                sample_id="reference-1",
                chapter="Contracts",
                section="Offer",
                question="What is an offer?",
                gold_answer="A proposal capable of acceptance.",
            ),
            ReferenceSample(
                schema_version=1,
                dataset_id="client-real-reference",
                dataset_version="v1",
                sample_id="reference-2",
                chapter="Contracts",
                section="Acceptance",
                question="What is acceptance?",
                gold_answer="Final assent to the offer.",
            ),
        ]
        reference_path = root / "reference.jsonl"
        write_reference_jsonl(reference_path, reference_samples)
        reference_identity = reference_dataset_identity(reference_samples)

        stack = Stack(root)
        stack.host_runtime.model_profile = pinned_host_profile()
        profile = pinned_client_profile(QWEN_PROFILE_ID)
        execution_profile = TrainingExecutionProfile(
            backend="transformers",
            device="cuda",
            precision="bfloat16",
            micro_batch_size=1,
            gradient_accumulation_steps=1,
        )
        runtime = ClientRuntime(
            data_dir=root / "client-a",
            client_id="client-a",
            model_profile=profile,
            private_data_path=private_path,
            training_execution_profile=execution_profile,
        )
        gateway = CoordinatorGateway(
            "http://coordinator",
            stack.issue_enrollment_token(),
            transport=stack.coordinator_transport,
        )
        app = create_client_app(
            runtime,
            gateway,
            admin_token_override=stack.client_admin_token,
        )
        runtime_b = ClientRuntime(
            data_dir=root / "client-b",
            client_id="client-b",
            model_profile=profile,
            training_execution_profile=execution_profile,
        )
        gateway_b = CoordinatorGateway(
            "http://coordinator",
            stack.issue_enrollment_token(),
            transport=stack.coordinator_transport,
        )
        app_b = create_client_app(
            runtime_b,
            gateway_b,
            admin_token_override=stack.client_admin_token,
        )
        for client_app in (app, app_b):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=client_app),
                base_url="http://client",
                headers=stack.client_headers,
            ) as client:
                response = await client.post("/v1/register")
                self.assertEqual(response.status_code, 201, response.text)

        create = await stack.coordinator_request(
            "POST",
            "/v1/rounds",
            headers={"X-Admin-Token": stack.admin_token},
            json={
                "selected_client_ids": ["client-a", "client-b"],
                "trusted_client_quorum": 2,
                "reference_dataset_id": reference_identity.dataset_id,
                "reference_dataset_hash": reference_identity.dataset_hash,
                "sample_ids": [item.sample_id for item in reference_samples],
                "prompt_template": PROMPT_TEMPLATE,
                "label_format": "chat_sft_answer_only_v1",
                "maximum_sequence_length": 2048,
                "truncation_policy": "reject",
                "top_k": 2,
                "training_epochs": 1,
            },
        )
        self.assertEqual(create.status_code, 201, create.text)
        manifest = RoundManifest.model_validate(create.json())
        runtime.cache_reference_dataset(
            manifest=manifest,
            content=reference_path.read_bytes(),
        )

        staging = runtime.adapter_store.staging_path("real-package-v1")
        self._write_adapter(staging, b"STEP36-ADAPTER-V1")
        metadata = runtime.adapter_store.seal(
            staging,
            version=1,
            parent=None,
            round_id=manifest.round_id,
            manifest_hash=manifest.manifest_hash,
            execution_profile_hash=execution_profile.profile_hash(),
        )
        runtime.adapter_store.promote(staging, metadata)
        examples = load_private_examples(private_path)
        record = LocalTrainingRecord.create(
            round_id=manifest.round_id,
            manifest_hash=manifest.manifest_hash,
            client_model_profile_hash=profile.profile_hash(),
            training_execution_profile=execution_profile,
            training_execution_profile_hash=execution_profile.profile_hash(),
            private_dataset_id=runtime.private_dataset_id,
            private_dataset_semantic_hash=private_dataset_semantic_hash(examples),
            private_example_count=len(examples),
            parent_adapter_version=None,
            parent_checkpoint_hash=None,
            result_adapter_version=metadata.version,
            result_checkpoint_hash=metadata.checkpoint_hash,
            checkpoint_format="peft-safetensors",
            label_format="chat_sft_answer_only_v1",
            maximum_sequence_length=manifest.maximum_sequence_length,
            truncation_policy="reject",
            started_at=utc_text(),
            completed_at=utc_text(),
            dependency_versions={"torch": "test-double"},
            trainable_parameter_count=8,
            total_parameter_count=100,
            optimizer_step_count=1,
            training_loss=1.0,
        )
        runtime.store.write_json(
            runtime._local_training_record_path(manifest.round_id),
            record.model_dump(mode="json"),
        )
        state = runtime.state()
        state.update(
            {
                "candidate_adapter_version": metadata.version,
                "training_adapter_version": metadata.version,
                "training_checkpoint_hash": metadata.checkpoint_hash,
                "local_training_runs": 1,
                "last_training_round": manifest.round_id,
            }
        )
        runtime.store.write_json("state.json", state)
        knowledge = self._knowledge(manifest)
        generator = mock.Mock(return_value=knowledge)
        runtime.generate_knowledge_samples = generator
        return {
            "stack": stack,
            "runtime": runtime,
            "app": app,
            "manifest": manifest,
            "record": record,
            "knowledge": knowledge,
            "generator": generator,
            "private_values": [private_marker, private_answer],
        }

    def _advance_adapter(self, runtime: ClientRuntime) -> None:
        parent, _ = runtime.adapter_store.current()
        staging = runtime.adapter_store.staging_path("real-package-v2")
        self._write_adapter(staging, b"STEP36-ADAPTER-V2")
        metadata = runtime.adapter_store.seal(
            staging,
            version=2,
            parent=parent,
            round_id="round-after-real-package",
            manifest_hash="0" * 64,
            execution_profile_hash=runtime.training_execution_profile.profile_hash(),
        )
        runtime.adapter_store.promote(staging, metadata)
        state = runtime.state()
        state.update(
            {
                "candidate_adapter_version": metadata.version,
                "training_adapter_version": metadata.version,
                "training_checkpoint_hash": metadata.checkpoint_hash,
                "last_training_round": "round-after-real-package",
            }
        )
        runtime.store.write_json("state.json", state)

    async def test_real_package_is_retryable_and_pinned_to_archived_adapter(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = await self._fixture(Path(directory))
            runtime = fixture["runtime"]
            manifest = fixture["manifest"]
            package = runtime.create_knowledge_package(manifest)
            artifact_path = runtime.package_artifact_path(package)
            package_bytes = runtime.store.path(
                runtime._pending_package_path(manifest.round_id)
            ).read_bytes()
            artifact_bytes = artifact_path.read_bytes()

            self.assertEqual(package.adapter_version, 1)
            self.assertEqual(package.sample_ids, manifest.sample_ids)
            self.assertEqual(
                load_package_samples(
                    artifact_path,
                    package,
                    maximum_bytes=manifest.maximum_knowledge_package_bytes,
                ),
                fixture["knowledge"],
            )
            exported = package_bytes + artifact_bytes
            for value in fixture["private_values"]:
                self.assertNotIn(value.encode("utf-8"), exported)

            self._advance_adapter(runtime)
            retry = runtime.create_knowledge_package(manifest)
            self.assertEqual(retry, package)
            self.assertEqual(
                runtime.store.path(
                    runtime._pending_package_path(manifest.round_id)
                ).read_bytes(),
                package_bytes,
            )
            self.assertEqual(
                runtime.package_artifact_path(retry).read_bytes(), artifact_bytes
            )
            self.assertEqual(fixture["generator"].call_count, 1)

            snapshot_path = runtime._pending_snapshot_path(manifest.round_id)
            snapshot = runtime.store.read_json(snapshot_path)
            tampered = {**snapshot, "artifact_sha256": "f" * 64}
            runtime.store.write_json(snapshot_path, tampered)
            with self.assertRaisesRegex(ClientRuntimeError, "snapshot differs"):
                runtime.create_knowledge_package(manifest)
            runtime.store.write_json(snapshot_path, snapshot)

            archived = runtime.adapter_store.version_path(1)
            (archived / "adapter_model.safetensors").write_bytes(
                b"TAMPERED-STEP36-ADAPTER-V1"
            )
            with self.assertRaisesRegex(ClientRuntimeError, "checkpoint is invalid"):
                runtime.create_knowledge_package(manifest)

    async def test_round_endpoint_runs_off_loop_locks_ml_and_persists_exact_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = await self._fixture(root)
            runtime = fixture["runtime"]
            manifest = fixture["manifest"]
            alignment_preflight = mock.Mock(return_value=None)
            runtime.ensure_alignment_tokenizers = alignment_preflight
            original_create = runtime.create_knowledge_package
            started = threading.Event()
            release = threading.Event()

            def slow_create(signed_manifest):
                started.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("test knowledge generation release timed out")
                return original_create(signed_manifest)

            runtime.create_knowledge_package = slow_create
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=fixture["app"]),
                base_url="http://client-a",
                headers=fixture["stack"].client_headers,
            ) as client:
                request = asyncio.create_task(
                    client.post(f"/v1/rounds/{manifest.round_id}/participate")
                )
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 2))
                    health = await client.get("/health")
                    concurrent = await client.post(
                        f"/v1/rounds/{manifest.round_id}/local-train"
                    )
                finally:
                    release.set()
                submitted = await asyncio.wait_for(request, timeout=5)

            self.assertEqual(health.status_code, 200, health.text)
            self.assertEqual(concurrent.status_code, 409, concurrent.text)
            self.assertIn("another local ML job", concurrent.text)
            self.assertEqual(submitted.status_code, 201, submitted.text)
            self.assertEqual(submitted.json()["state"], "COLLECTING")
            self.assertEqual(submitted.json()["accepted_count"], 1)
            self.assertEqual(submitted.json()["quorum"], 2)

            client_package_path = runtime.store.path(
                runtime._accepted_package_path(manifest.round_id)
            )
            client_artifact_path = runtime.store.path(
                runtime._accepted_artifact_path(manifest.round_id)
            )
            coordinator_package_path = (
                root
                / "coordinator"
                / "rounds"
                / manifest.round_id
                / "submissions"
                / "client-a"
                / "package.json"
            )
            coordinator_artifact_path = coordinator_package_path.with_name(
                "knowledge.safetensors"
            )
            client_package = KnowledgePackage.model_validate_json(
                client_package_path.read_text(encoding="utf-8")
            )
            coordinator_package = KnowledgePackage.model_validate_json(
                coordinator_package_path.read_text(encoding="utf-8")
            )
            self.assertEqual(client_package, coordinator_package)
            self.assertEqual(
                client_artifact_path.read_bytes(),
                coordinator_artifact_path.read_bytes(),
            )
            self.assertEqual(fixture["generator"].call_count, 1)
            self.assertEqual(alignment_preflight.call_count, 1)
            state = await fixture["stack"].coordinator_service.round_status(
                manifest.round_id
            )
            self.assertEqual(state.state, "COLLECTING")
            self.assertEqual(state.accepted_client_ids, ["client-a"])
            self.assertFalse(
                (
                    root
                    / "coordinator"
                    / "rounds"
                    / manifest.round_id
                    / "submissions"
                    / "client-b"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
