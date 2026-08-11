from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import httpx
from pydantic import ValidationError

from client.main import CoordinatorGateway, create_app as create_client_app
from client.model_profiles import (
    LLAMA_PROFILE_ID,
    LLAMA_REVISION,
    QWEN_PROFILE_ID,
    QWEN_REVISION,
    pinned_client_profile,
)
from client.runtime import ClientRuntime, ClientRuntimeError
from client.training import (
    AdapterCheckpointStore,
    PrivateTrainingExample,
    TrainingExecutionProfile,
    encode_private_examples,
    load_private_examples,
    private_dataset_semantic_hash,
)
from host.runtime import default_host_profile
from shared.crypto import Ed25519Identity, sha256_hex
from shared.prompt import PROMPT_TEMPLATE, PROMPT_TEMPLATE_ID
from shared.protocol import LoraProfile, ModelProfile, RoundCreateRequest, RoundManifest
from shared.protocol import utc_now, utc_text
from tests.test_round import Stack


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.options: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        **options,
    ):
        self.options.append(options)
        prompt = messages[0]["content"]
        prefix = [11, len(prompt), 12]
        if add_generation_prompt:
            return prefix
        answer = messages[1]["content"]
        return prefix + [21, len(answer), 22]


def mock_profile() -> ModelProfile:
    return ModelProfile(
        profile_id="client-mock-training-v1",
        role="client",
        model_id="legalfedllm/mock-client",
        model_revision="mock-v1",
        tokenizer_id="legalfedllm/mock-tokenizer",
        tokenizer_revision="mock-v1",
        tokenizer_class="MockTokenizer",
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_hash=sha256_hex(PROMPT_TEMPLATE.encode("utf-8")),
        lora=LoraProfile(rank=8),
    )


def manifest_for(
    directory: str,
    profile: ModelProfile,
    *,
    round_id: str = "round-training",
) -> RoundManifest:
    identity = Ed25519Identity.load_or_create(Path(directory) / "coordinator.pem")
    request = RoundCreateRequest(
        selected_client_ids=["client-a"],
        trusted_client_quorum=1,
        reference_dataset_id="reference-v1",
        reference_dataset_hash=sha256_hex(b"reference-v1"),
        sample_ids=["sample-1"],
        prompt_template=PROMPT_TEMPLATE,
        label_format="chat_sft_answer_only_v1",
        maximum_sequence_length=512,
        truncation_policy="reject",
        top_k=3,
        training_epochs=1,
    )
    return RoundManifest.create_signed(
        identity=identity,
        round_id=round_id,
        coordinator_id="coordinator",
        current_host_adapter_version=0,
        host_model_profile=default_host_profile(),
        selected_client_profile_hashes={"client-a": profile.profile_hash()},
        request=request,
        submission_deadline=utc_text(utc_now() + timedelta(hours=1)),
    )


class ClientModelProfileTests(unittest.TestCase):
    def test_pinned_profiles_are_exact_and_share_the_approved_lora(self) -> None:
        llama = pinned_client_profile(LLAMA_PROFILE_ID)
        qwen = pinned_client_profile(QWEN_PROFILE_ID)
        self.assertEqual(llama.model_revision, LLAMA_REVISION)
        self.assertEqual(qwen.model_revision, QWEN_REVISION)
        self.assertEqual(llama.model_revision, llama.tokenizer_revision)
        self.assertEqual(qwen.model_revision, qwen.tokenizer_revision)
        self.assertEqual(qwen.chat_template_mode, "qwen_non_thinking")
        self.assertEqual(
            qwen.profile_hash(),
            pinned_client_profile(
                QWEN_PROFILE_ID,
                serving_backend="ollama",
            ).profile_hash(),
        )
        self.assertEqual(
            llama.prompt_template_hash,
            sha256_hex(PROMPT_TEMPLATE.encode("utf-8")),
        )
        for profile in (llama, qwen):
            self.assertEqual(profile.training_backend, "transformers")
            self.assertEqual(profile.lora.rank, 8)
            self.assertEqual(profile.lora.alpha, 16)
            self.assertEqual(profile.lora.dropout, 0.05)
            self.assertEqual(
                profile.lora.target_modules,
                ("q_proj", "k_proj", "v_proj", "o_proj"),
            )
            self.assertEqual(profile.lora.modules_to_save, ())

    def test_transformers_profile_rejects_a_moving_revision(self) -> None:
        values = pinned_client_profile(QWEN_PROFILE_ID).model_dump(mode="json")
        values["model_revision"] = "main"
        with self.assertRaises(ValidationError):
            ModelProfile.model_validate(values)


class PrivateTrainingContractTests(unittest.TestCase):
    def test_jsonl_is_strict_ordered_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            rows = [
                {
                    "schema_version": "1.0",
                    "example_id": "private-1",
                    "prompt": "  retained prompt  ",
                    "answer": "retained answer",
                },
                {
                    "schema_version": "1.0",
                    "example_id": "private-2",
                    "prompt": "second prompt",
                    "answer": "second answer",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            examples = load_private_examples(path)
            self.assertEqual(examples[0].prompt, "  retained prompt  ")
            original_hash = private_dataset_semantic_hash(examples)
            reversed_hash = private_dataset_semantic_hash(list(reversed(examples)))
            self.assertNotEqual(original_hash, reversed_hash)
            self.assertNotIn("retained prompt", original_hash)

    def test_jsonl_rejects_extra_fields_and_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(
                '{"schema_version":"1.0","example_id":"x",'
                '"prompt":"p","answer":"a","source":"private"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source"):
                load_private_examples(path)
            path.write_text(
                '{"schema_version":"1.0","example_id":"x",'
                '"prompt":"p","answer":"a"}\n'
                '{"schema_version":"1.0","example_id":"x",'
                '"prompt":"p2","answer":"a2"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_private_examples(path)

    def test_answer_only_labels_and_qwen_non_thinking_mode(self) -> None:
        tokenizer = FakeTokenizer()
        examples = [
            PrivateTrainingExample(
                example_id="private-1",
                prompt="question",
                answer="answer",
            )
        ]
        encoded = encode_private_examples(
            examples,
            tokenizer=tokenizer,
            model_profile=pinned_client_profile(QWEN_PROFILE_ID),
            maximum_sequence_length=32,
        )
        self.assertEqual(encoded[0].labels[:3], [-100, -100, -100])
        self.assertEqual(encoded[0].labels[3:], [21, 6, 22])
        self.assertEqual(encoded[0].attention_mask, [1] * 6)
        self.assertEqual(
            tokenizer.options,
            [{"enable_thinking": False}, {"enable_thinking": False}],
        )

    def test_overlength_examples_are_rejected_not_truncated(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeding"):
            encode_private_examples(
                [
                    PrivateTrainingExample(
                        example_id="private-1",
                        prompt="question",
                        answer="answer",
                    )
                ],
                tokenizer=FakeTokenizer(),
                model_profile=pinned_client_profile(LLAMA_PROFILE_ID),
                maximum_sequence_length=5,
            )

    def test_execution_profile_rejects_silent_cpu_mixed_precision(self) -> None:
        with self.assertRaises(ValidationError):
            TrainingExecutionProfile(
                backend="transformers",
                device="cpu",
                precision="bfloat16",
            )


class AdapterCheckpointStoreTests(unittest.TestCase):
    @staticmethod
    def write_adapter(path: Path, payload: bytes) -> None:
        (path / "adapter_config.json").write_text(
            '{"peft_type":"LORA"}\n',
            encoding="utf-8",
        )
        (path / "adapter_model.safetensors").write_bytes(payload)

    def test_atomic_pointer_ignores_incomplete_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = pinned_client_profile(QWEN_PROFILE_ID)
            store = AdapterCheckpointStore(directory, profile)
            initial = store.staging_path("initial")
            self.write_adapter(initial, b"initial")
            metadata = store.seal(
                initial,
                version=0,
                parent=None,
                round_id=None,
                manifest_hash=None,
                execution_profile_hash=None,
            )
            store.promote(initial, metadata)
            incomplete = store.staging_path("interrupted")
            (incomplete / "adapter_config.json").write_text(
                "{}\n", encoding="utf-8"
            )
            current, _ = store.current()
            self.assertEqual(current.version, 0)
            self.assertEqual(current.checkpoint_hash, metadata.checkpoint_hash)
            serving_store = AdapterCheckpointStore(
                directory,
                pinned_client_profile(
                    QWEN_PROFILE_ID,
                    serving_backend="ollama",
                ),
            )
            serving_current, _ = serving_store.current()
            self.assertEqual(serving_current.checkpoint_hash, metadata.checkpoint_hash)
            orphan = store.version_path(1)
            orphan.mkdir()
            self.assertEqual(store.next_version(1), 2)
            current, _ = store.current()
            self.assertEqual(current.version, 0)

    def test_current_checkpoint_rejects_tampering_and_profile_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = pinned_client_profile(QWEN_PROFILE_ID)
            store = AdapterCheckpointStore(directory, profile)
            initial = store.staging_path("initial")
            self.write_adapter(initial, b"initial")
            metadata = store.seal(
                initial,
                version=0,
                parent=None,
                round_id=None,
                manifest_hash=None,
                execution_profile_hash=None,
            )
            path = store.promote(initial, metadata)
            incompatible_profile = profile.model_copy(
                update={"lora": profile.lora.model_copy(update={"rank": 4})}
            )
            incompatible_store = AdapterCheckpointStore(
                directory,
                incompatible_profile,
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                incompatible_store.current()
            (path / "adapter_model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash"):
                store.current()


class RoundBoundTrainingTests(unittest.TestCase):
    def test_round_record_binds_manifest_dataset_profile_and_adapter(self) -> None:
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("peft", sys.modules)
        with tempfile.TemporaryDirectory() as directory:
            private_path = Path(directory) / "private.jsonl"
            private_marker = "confidential matter alpha"
            private_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "example_id": "private-1",
                        "prompt": private_marker,
                        "answer": "reviewed answer",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            profile = mock_profile()
            runtime = ClientRuntime(
                data_dir=Path(directory) / "client",
                client_id="client-a",
                model_profile=profile,
                private_data_path=private_path,
            )
            first_manifest = manifest_for(directory, profile, round_id="round-one")
            record = runtime.local_train_round(first_manifest)
            self.assertNotIn(private_marker, json.dumps(record))
            self.assertEqual(record["round_id"], "round-one")
            self.assertEqual(record["result_adapter_version"], 1)
            runtime.require_round_training(first_manifest)

            changed = json.loads(private_path.read_text(encoding="utf-8"))
            changed["answer"] = "changed after training"
            private_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ClientRuntimeError, "dataset changed"):
                runtime.require_round_training(first_manifest)
            changed["answer"] = "reviewed answer"
            private_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")

            second_manifest = manifest_for(directory, profile, round_id="round-two")
            runtime.local_train_round(second_manifest)
            with self.assertRaisesRegex(ClientRuntimeError, "another local"):
                runtime.require_round_training(first_manifest)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("peft", sys.modules)


class RoundBoundTrainingApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_specific_training_is_required_before_participation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_path = root / "private.jsonl"
            private_marker = "confidential strict endpoint example"
            private_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "example_id": "private-1",
                        "prompt": private_marker,
                        "answer": "reviewed answer",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stack = Stack(directory)
            runtime = ClientRuntime(
                data_dir=root / "client-a",
                client_id="client-a",
                private_data_path=private_path,
            )
            gateway = CoordinatorGateway(
                "http://coordinator",
                stack.registration_token,
                transport=stack.coordinator_transport,
            )
            app = create_client_app(
                runtime,
                gateway,
                admin_token_override=stack.client_admin_token,
            )
            headers = stack.client_headers
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client",
                headers=headers,
            ) as client:
                self.assertEqual((await client.post("/v1/register")).status_code, 201)

            create = await stack.coordinator_request(
                "POST",
                "/v1/rounds",
                headers={"X-Admin-Token": stack.admin_token},
                json={
                    "selected_client_ids": ["client-a"],
                    "trusted_client_quorum": 1,
                    "reference_dataset_id": "reference-v1",
                    "reference_dataset_hash": sha256_hex(b"reference-v1"),
                    "sample_ids": ["sample-1"],
                    "prompt_template": PROMPT_TEMPLATE,
                    "label_format": "chat_sft_answer_only_v1",
                    "maximum_sequence_length": 512,
                    "truncation_policy": "reject",
                    "top_k": 3,
                },
            )
            self.assertEqual(create.status_code, 201, create.text)
            manifest = RoundManifest.model_validate(create.json())
            self.assertEqual(
                manifest.selected_client_profile_hashes["client-a"],
                runtime.model_profile.profile_hash(),
            )

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client",
                headers=headers,
            ) as client:
                blocked = await client.post(
                    f"/v1/rounds/{manifest.round_id}/participate"
                )
                trained = await client.post(
                    f"/v1/rounds/{manifest.round_id}/local-train"
                )
                participated = await client.post(
                    f"/v1/rounds/{manifest.round_id}/participate"
                )
            self.assertEqual(blocked.status_code, 409, blocked.text)
            self.assertEqual(trained.status_code, 200, trained.text)
            self.assertNotIn(private_marker, trained.text)
            self.assertEqual(participated.status_code, 201, participated.text)
            package_path = (
                root
                / "coordinator"
                / "rounds"
                / manifest.round_id
                / "submissions"
                / "client-a"
                / "package.json"
            )
            self.assertNotIn(
                private_marker,
                package_path.read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                private_marker.encode("utf-8"),
                package_path.with_name("knowledge.safetensors").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
