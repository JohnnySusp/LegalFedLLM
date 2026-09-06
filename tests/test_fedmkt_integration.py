from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

try:
    if not TORCH_AVAILABLE:
        raise ModuleNotFoundError("torch")
    from shared.alignment_profiles import (
        BidirectionalAlignmentProfile,
        TokenizerEndpoint,
    )
    from shared.crypto import sha256_hex
    from shared.distillation_artifact import (
        load_host_training_artifact,
        write_host_training_artifact,
    )
    from shared.fedmkt_core.integration import (
        DistillationIntegrationError,
        TrustedClientQuorumError,
        integrate_distillation_round,
    )
    from shared.fedmkt_core.safety import (
        MINIMUM_TRUST_SCORE,
        inspect_knowledge_package,
    )
    from shared.fedmkt_core.ml.sparse_targets import (
        answer_only_sparse_distillation_loss,
    )
    from shared.fedmkt_core.ml.token_alignment import transform_step_logits
    from shared.protocol import (
        KnowledgePackage,
        KnowledgeSample,
        LoraProfile,
        ModelProfile,
        SafetyReport,
    )
    from shared.tokenizer_validation import ValidatedTokenizer
    from shared.vocabulary_mapping import VocabularyMappingCache
except ModuleNotFoundError as exc:
    INTEGRATION_IMPORT_ERROR = exc
else:
    INTEGRATION_IMPORT_ERROR = None


HASH = "0" * 64


class FakeTokenizer:
    def __init__(self, vocabulary: dict[str, int]) -> None:
        self.vocabulary = vocabulary
        self.tokens = {token_id: token for token, token_id in vocabulary.items()}

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocabulary)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str | None]:
        return [self.tokens.get(token_id) for token_id in token_ids]


def endpoint(
    *,
    role: str,
    profile_id: str,
    marker: str,
    artifact_hash: str,
    vocabulary_size: int,
    tokenizer_vocabulary_size: int,
    pad_token_id: int,
) -> TokenizerEndpoint:
    return TokenizerEndpoint(
        role=role,
        profile_id=profile_id,
        model_id=f"test/{profile_id}",
        model_revision="revision",
        model_class="FakeForCausalLM",
        model_type="fake",
        tokenizer_id=f"test/{profile_id}",
        tokenizer_revision="revision",
        tokenizer_class="FakeTokenizer",
        vocabulary_size=vocabulary_size,
        tokenizer_chat_template_hash="a" * 64,
        chat_template_mode="standard",
        tokenizer_artifact_sha256=artifact_hash,
        tokenizer_base_vocabulary_size=tokenizer_vocabulary_size,
        tokenizer_vocabulary_size=tokenizer_vocabulary_size,
        tokenizer_max_token_id=tokenizer_vocabulary_size - 1,
        word_boundary_marker=marker,
        bos_token=None,
        bos_token_id=None,
        eos_token=None,
        eos_token_id=None,
        pad_token="<pad>",
        pad_token_id=pad_token_id,
        unk_token=None,
        unk_token_id=None,
        additional_special_token_ids=(),
        model_max_length=128,
        padding_side="right",
    )


def model_profile(value: TokenizerEndpoint) -> ModelProfile:
    return ModelProfile(
        role=value.role,
        profile_id=value.profile_id,
        model_id=value.model_id,
        model_revision=value.model_revision,
        model_class=value.model_class,
        model_type=value.model_type,
        tokenizer_id=value.tokenizer_id,
        tokenizer_revision=value.tokenizer_revision,
        tokenizer_class=value.tokenizer_class,
        vocabulary_size=value.vocabulary_size,
        prompt_template_hash=HASH,
        tokenizer_chat_template_hash=value.tokenizer_chat_template_hash,
        chat_template_mode=value.chat_template_mode,
        lora=LoraProfile(rank=8),
    )


def package(
    sender_id: str,
    role: str,
    profile: TokenizerEndpoint,
    sample_ids: list[str],
    alignment_profile_id: str,
) -> KnowledgePackage:
    return KnowledgePackage.model_construct(
        round_id="round-1",
        manifest_hash=HASH,
        sender_id=sender_id,
        sender_role=role,
        model_profile=model_profile(profile),
        adapter_version=3 if role == "host" else 1,
        alignment_profile_id=alignment_profile_id,
        reference_dataset_id="reference-1",
        reference_dataset_hash="1" * 64,
        sample_ids=sample_ids,
        top_k=2,
        package_hash=sha256_hex(sender_id),
    )


def sample(
    sample_id: str,
    ce_loss: float,
    *,
    source_ids: list[int],
    top_k_ids: list[list[int]],
    logit_offset: float = 0.0,
) -> KnowledgeSample:
    return KnowledgeSample(
        sample_id=sample_id,
        source_input_ids=source_ids,
        attention_length=len(source_ids),
        top_k_token_ids=top_k_ids,
        top_k_logits=[
            [3.0 + logit_offset, 0.0 + logit_offset]
            for _ in source_ids
        ],
        full_logsumexp=[
            3.0 + logit_offset + ce_loss,
            *([5.0 + logit_offset] * (len(source_ids) - 1)),
        ],
        gold_token_ids=[top_k_ids[0][0], *([-100] * (len(source_ids) - 1))],
        gold_token_logits=[
            3.0 + logit_offset,
            *([0.0] * (len(source_ids) - 1)),
        ],
        gold_token_nll=[ce_loss, *([0.0] * (len(source_ids) - 1))],
        ce_loss=ce_loss,
    )


@unittest.skipUnless(
    TORCH_AVAILABLE and INTEGRATION_IMPORT_ERROR is None,
    f"optional ML dependencies are unavailable: {INTEGRATION_IMPORT_ERROR}",
)
class FedMKTIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.host_endpoint = endpoint(
            role="host",
            profile_id="host-test",
            marker="▁",
            artifact_hash="2" * 64,
            vocabulary_size=8,
            tokenizer_vocabulary_size=8,
            pad_token_id=0,
        )
        self.client_endpoint = endpoint(
            role="client",
            profile_id="client-test",
            marker="Ġ",
            artifact_hash="1" * 64,
            vocabulary_size=18,
            tokenizer_vocabulary_size=18,
            pad_token_id=0,
        )
        self.profile = BidirectionalAlignmentProfile(
            profile_id="dtw:test-v1",
            strategy="dtw",
            profile_version="test-v1",
            client=self.client_endpoint,
            host=self.host_endpoint,
            client_to_host_owner="coordinator",
            host_to_client_owner="client",
        )
        self.granite_client_endpoint = endpoint(
            role="client",
            profile_id="granite-client-test",
            marker="▁",
            artifact_hash="3" * 64,
            vocabulary_size=8,
            tokenizer_vocabulary_size=8,
            pad_token_id=0,
        )
        self.granite_profile = BidirectionalAlignmentProfile(
            profile_id="dtw:granite-test-v1",
            strategy="dtw",
            profile_version="granite-test-v1",
            client=self.granite_client_endpoint,
            host=self.host_endpoint,
            client_to_host_owner="coordinator",
            host_to_client_owner="client",
        )
        self.host_tokenizer = ValidatedTokenizer(
            endpoint=self.host_endpoint,
            tokenizer=FakeTokenizer(
                {
                    "<pad>": 0,
                    "▁a": 1,
                    "▁b": 2,
                    "▁c": 3,
                    "▁d": 4,
                    "▁x": 5,
                    "▁y": 6,
                    "▁z": 7,
                }
            ),
            artifact_sha256=self.host_endpoint.tokenizer_artifact_sha256,
        )
        self.client_tokenizer = ValidatedTokenizer(
            endpoint=self.client_endpoint,
            tokenizer=FakeTokenizer(
                {
                    **{f"unused-{index}": index for index in range(10)},
                    "Ġa": 10,
                    "Ġb": 11,
                    "Ġc": 12,
                    "Ġd": 13,
                    "Ġx": 14,
                    "Ġy": 15,
                    "Ġz": 16,
                    "<pad>": 17,
                }
            ),
            artifact_sha256=self.client_endpoint.tokenizer_artifact_sha256,
        )
        self.granite_client_tokenizer = ValidatedTokenizer(
            endpoint=self.granite_client_endpoint,
            tokenizer=FakeTokenizer(
                {
                    "<pad>": 0,
                    "▁a": 1,
                    "▁b": 2,
                    "▁c": 3,
                    "▁d": 4,
                    "▁x": 5,
                    "▁y": 6,
                    "▁z": 7,
                }
            ),
            artifact_sha256=(
                self.granite_client_endpoint.tokenizer_artifact_sha256
            ),
        )
        self.sample_ids = [
            "host-best",
            "host-tie",
            "client-tie",
            "client-best",
        ]
        self.host_package = package(
            "host",
            "host",
            self.host_endpoint,
            self.sample_ids,
            self.profile.profile_id,
        )
        self.host_samples = [
            sample(
                "host-best",
                0.10,
                source_ids=[1, 2, 3],
                top_k_ids=[[1, 4], [2, 4], [3, 4]],
            ),
            sample(
                "host-tie",
                0.20,
                source_ids=[1, 2, 3],
                top_k_ids=[[1, 4], [2, 4], [3, 4]],
            ),
            sample(
                "client-tie",
                0.50,
                source_ids=[1, 2, 3],
                top_k_ids=[[1, 4], [2, 4], [3, 4]],
            ),
            sample(
                "client-best",
                0.50,
                source_ids=[1, 2, 3],
                top_k_ids=[[1, 4], [2, 4], [3, 4]],
            ),
        ]
        losses = {
            "client-b": [0.30, 0.20, 0.25, 0.20],
            "client-a": [0.40, 0.35, 0.25, 0.10],
            "below": [0.01, 0.01, 0.01, 0.01],
            "hard-rejected": [0.01, 0.01, 0.01, 0.01],
        }
        self.client_packages = [
            package(
                client_id,
                "client",
                self.client_endpoint,
                self.sample_ids,
                self.profile.profile_id,
            )
            for client_id in losses
        ]
        self.client_samples = {
            client_id: [
                sample(
                    sample_id,
                    ce_loss,
                    source_ids=[10, 11, 12],
                    top_k_ids=[[10, 13], [11, 13], [12, 13]],
                    logit_offset=index / 10,
                )
                for index, (sample_id, ce_loss) in enumerate(
                    zip(self.sample_ids, values)
                )
            ]
            for client_id, values in losses.items()
        }
        self.safety_reports = {
            "client-b": SafetyReport(accepted=True, trust_score=0.5),
            "client-a": SafetyReport(accepted=True, trust_score=1.0),
            "below": SafetyReport(
                accepted=True,
                trust_score=round(MINIMUM_TRUST_SCORE - 0.01, 2),
            ),
            "hard-rejected": SafetyReport(
                accepted=False,
                trust_score=0.99,
                reasons=["hard protocol check failed"],
            ),
        }
        self.selected_client_ids = [
            "client-b",
            "below",
            "client-a",
            "hard-rejected",
        ]
        self.labels = {
            sample_id: [-100, 2, 3] for sample_id in self.sample_ids
        }

    def integrate(self, directory: str, **changes):
        values = {
            "profile": self.profile,
            "client_tokenizer": self.client_tokenizer,
            "host_tokenizer": self.host_tokenizer,
            "mapping_cache": VocabularyMappingCache(directory),
            "host_package": self.host_package,
            "host_samples": self.host_samples,
            "client_packages": self.client_packages,
            "client_samples": self.client_samples,
            "safety_reports": self.safety_reports,
            "selected_client_ids": self.selected_client_ids,
            "trusted_client_quorum": 2,
            "labels_by_sample": self.labels,
            "temperature": 1.0,
            "loss_type": "ce",
        }
        values.update(changes)
        return integrate_distillation_round(**values)

    def test_pre_alignment_reports_are_finalized_before_teacher_selection(self) -> None:
        selected = ["client-a"]
        package_by_id = {item.sender_id: item for item in self.client_packages}
        reports = {
            client_id: inspect_knowledge_package(
                package_by_id[client_id],
                self.client_samples[client_id],
            )
            for client_id in selected
        }
        self.assertEqual(reports["client-a"].probe_stage, "pre_alignment")

        with tempfile.TemporaryDirectory() as directory:
            batch = self.integrate(
                directory,
                client_packages=[package_by_id["client-a"]],
                client_samples={"client-a": self.client_samples["client-a"]},
                safety_reports=reports,
                selected_client_ids=selected,
                trusted_client_quorum=1,
                historical_reliability={"client-a": 0.5},
            )

        final = batch.audit.safety_reports["client-a"]
        self.assertEqual(final.probe_stage, "post_alignment")
        self.assertEqual(final.score_components["peer_consistency"], 0.5)
        self.assertGreaterEqual(final.trust_score, 0.5)
        self.assertEqual(batch.dataset.accepted_client_ids, ["client-a"])

    def test_validated_packages_reach_deterministic_sparse_trainer_inputs(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            first = self.integrate(directory)
            second = self.integrate(directory)

        self.assertEqual(first.dataset.accepted_client_ids, ["client-b", "client-a"])
        self.assertEqual(
            [sample.teacher_id for sample in first.dataset.samples],
            ["host", "host", "client-b", "client-a"],
        )
        self.assertEqual(first.dataset.samples[2].trust_score, 0.5)
        self.assertEqual(first.dataset.dataset_hash, second.dataset.dataset_hash)
        self.assertEqual(first.audit.audit_hash, second.audit.audit_hash)
        self.assertEqual(first.audit.trainer_inputs_sha256, second.audit.trainer_inputs_sha256)
        self.assertEqual(
            [item.sender_id for item in first.audit.source_packages],
            ["host", "client-b", "client-a"],
        )
        self.assertEqual(
            [item.client_id for item in first.audit.rejected_clients],
            ["below", "hard-rejected"],
        )
        self.assertEqual(first.sparse_targets.probabilities.dtype, torch.float32)
        self.assertEqual(first.sparse_targets.probabilities.device.type, "cpu")
        self.assertEqual(first.input_ids.device.type, "cpu")
        self.assertEqual(
            set(first.trainer_inputs()),
            {
                "input_ids",
                "attention_mask",
                "labels",
                "sparse_target_token_ids",
                "sparse_target_probabilities",
                "sparse_target_valid_mask",
            },
        )

        model_logits = torch.zeros(4, 3, self.host_endpoint.vocabulary_size)
        loss = answer_only_sparse_distillation_loss(
            model_logits,
            first.sparse_targets,
            labels=first.labels,
            attention_mask=first.attention_mask,
            loss_type="ce",
        )
        torch.testing.assert_close(
            loss,
            torch.tensor(math.log(self.host_endpoint.vocabulary_size)),
        )

    def test_sparse_trainer_inputs_round_trip_through_safetensors(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            result = self.integrate(directory)
            first_path = Path(directory) / "first.safetensors"
            second_path = Path(directory) / "second.safetensors"
            descriptor = write_host_training_artifact(
                first_path,
                result,
                pad_token_id=self.host_endpoint.pad_token_id or 0,
                maximum_bytes=1024 * 1024,
            )
            second_descriptor = write_host_training_artifact(
                second_path,
                result,
                pad_token_id=self.host_endpoint.pad_token_id or 0,
                maximum_bytes=1024 * 1024,
            )
            tensors = load_host_training_artifact(
                first_path,
                descriptor,
                result.sample_ids,
                maximum_bytes=1024 * 1024,
                vocabulary_size=self.host_endpoint.vocabulary_size,
            )

            self.assertEqual(descriptor, second_descriptor)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertEqual(
                descriptor.trainer_inputs_sha256,
                result.audit.trainer_inputs_sha256,
            )
            self.assertEqual(tensors["input_ids"].dtype, np.dtype("int32"))
            self.assertEqual(tensors["labels"].dtype, np.dtype("int32"))
            self.assertEqual(
                tensors["sparse_target_token_ids"].dtype,
                np.dtype("int32"),
            )
            for name, value in result.trainer_inputs().items():
                expected = value.detach().cpu().numpy()
                if name in {"attention_mask", "sparse_target_valid_mask"}:
                    expected = expected.astype(np.uint8)
                elif name != "sparse_target_probabilities":
                    expected = expected.astype(np.int32)
                np.testing.assert_array_equal(tensors[name], expected)

            tampered = descriptor.model_copy(
                update={"trainer_inputs_sha256": "f" * 64}
            )
            with self.assertRaisesRegex(ValueError, "semantic tensor hash"):
                load_host_training_artifact(
                    first_path,
                    tampered,
                    result.sample_ids,
                    maximum_bytes=1024 * 1024,
                    vocabulary_size=self.host_endpoint.vocabulary_size,
                )
            oversized_path = Path(directory) / "oversized.safetensors"
            with self.assertRaisesRegex(ValueError, "tensors exceed"):
                write_host_training_artifact(
                    oversized_path,
                    result,
                    pad_token_id=self.host_endpoint.pad_token_id or 0,
                    maximum_bytes=128,
                )
            self.assertFalse(oversized_path.exists())

    def test_coordinator_freezes_one_immutable_host_training_job(self) -> None:
        from datetime import datetime, timedelta, timezone
        from types import SimpleNamespace
        from unittest import mock

        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        from coordinator.service import CoordinatorService, HostGateway
        from shared.crypto import Ed25519Identity
        from shared.prompt import PROMPT_TEMPLATE
        from shared.protocol import (
            RoundCreateRequest,
            RoundManifest,
        )

        request = RoundCreateRequest(
            selected_client_ids=list(self.selected_client_ids),
            trusted_client_quorum=2,
            reference_dataset_id="reference-1",
            reference_dataset_hash="1" * 64,
            sample_ids=list(self.sample_ids),
            prompt_template=PROMPT_TEMPLATE,
            label_format="causal_lm",
            maximum_sequence_length=128,
            truncation_policy="reject",
            top_k=2,
            host_public_data_epochs=1,
            maximum_host_training_job_bytes=1024 * 1024,
        )
        manifest = RoundManifest.create_signed(
            identity=Ed25519Identity(Ed25519PrivateKey.generate()),
            round_id="round-1",
            coordinator_id="coordinator",
            current_host_adapter_version=3,
            host_model_profile=model_profile(self.host_endpoint),
            selected_client_profile_hashes={
                client_id: "b" * 64 for client_id in self.selected_client_ids
            },
            selected_client_alignment_profiles={
                client_id: "dtw:test-v1"
                for client_id in self.selected_client_ids
            },
            request=request,
            submission_deadline=(
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat().replace("+00:00", "Z"),
        )

        with tempfile.TemporaryDirectory() as directory:
            service = CoordinatorService(
                data_dir=directory,
                host_gateway=HostGateway("http://host", "test-token"),
            )
            reference_bundle = SimpleNamespace(reference_samples=[object()])
            encoded = [
                SimpleNamespace(sample_id=sample_id, labels=self.labels[sample_id])
                for sample_id in self.sample_ids
            ]
            with (
                mock.patch(
                    "coordinator.service.resolve_alignment_profile",
                    return_value=self.profile,
                ),
                mock.patch(
                    "coordinator.service.load_pinned_tokenizer",
                    side_effect=[self.host_tokenizer, self.client_tokenizer],
                ),
                mock.patch.object(
                    service,
                    "_host_reference_dataset_bundle",
                    return_value=reference_bundle,
                ),
                mock.patch(
                    "coordinator.service.encode_reference_samples",
                    return_value=encoded,
                ),
            ):
                first = service._prepare_host_training_job(  # noqa: SLF001
                    manifest=manifest,
                    baseline=self.host_package,
                    baseline_samples=self.host_samples,
                    packages=self.client_packages,
                    client_samples=self.client_samples,
                    reports=self.safety_reports,
                )
                second = service._prepare_host_training_job(  # noqa: SLF001
                    manifest=manifest,
                    baseline=self.host_package,
                    baseline_samples=self.host_samples,
                    packages=self.client_packages,
                    client_samples=self.client_samples,
                    reports=self.safety_reports,
                )

            self.assertEqual(first, second)
            self.assertEqual(first.accepted_client_ids, ["client-b", "client-a"])
            self.assertEqual(first.host_public_data_epochs, 1)
            self.assertEqual(first.sample_ids, self.sample_ids)
            self.assertEqual(
                first.artifact.trainer_inputs_sha256,
                service.store.read_json(
                    "rounds/round-1/host_training_job/integration_audit.json"
                )["trainer_inputs_sha256"],
            )
            self.assertTrue(
                service.store.path(
                    "rounds/round-1/host_training_job/"
                    "trainer_inputs.safetensors"
                ).is_file()
            )
            self.assertFalse(
                service.store.exists(
                    "rounds/round-1/validated_distillation_dataset.json"
                )
            )

    def test_coordinator_prepares_mixed_alignment_host_training_job(self) -> None:
        from datetime import datetime, timedelta, timezone
        from types import SimpleNamespace
        from unittest import mock

        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        from coordinator.service import CoordinatorService, HostGateway
        from shared.crypto import Ed25519Identity
        from shared.prompt import PROMPT_TEMPLATE
        from shared.protocol import RoundCreateRequest, RoundManifest

        selected = ["client-b", "client-a"]
        request = RoundCreateRequest(
            selected_client_ids=selected,
            trusted_client_quorum=2,
            reference_dataset_id="reference-1",
            reference_dataset_hash="1" * 64,
            sample_ids=list(self.sample_ids),
            prompt_template=PROMPT_TEMPLATE,
            label_format="causal_lm",
            maximum_sequence_length=128,
            truncation_policy="reject",
            top_k=2,
            host_public_data_epochs=1,
            maximum_host_training_job_bytes=1024 * 1024,
        )
        manifest = RoundManifest.create_signed(
            identity=Ed25519Identity(Ed25519PrivateKey.generate()),
            round_id="round-mixed",
            coordinator_id="coordinator",
            current_host_adapter_version=3,
            host_model_profile=model_profile(self.host_endpoint),
            selected_client_profile_hashes={
                "client-b": model_profile(self.client_endpoint).profile_hash(),
                "client-a": model_profile(
                    self.granite_client_endpoint
                ).profile_hash(),
            },
            selected_client_alignment_profiles={
                "client-b": self.profile.profile_id,
                "client-a": self.granite_profile.profile_id,
            },
            request=request,
            submission_deadline=(
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat().replace("+00:00", "Z"),
        )
        mixed_packages = [
            package(
                "client-b",
                "client",
                self.client_endpoint,
                self.sample_ids,
                self.profile.profile_id,
            ),
            package(
                "client-a",
                "client",
                self.granite_client_endpoint,
                self.sample_ids,
                self.granite_profile.profile_id,
            ),
        ]
        mixed_samples = {
            "client-b": self.client_samples["client-b"],
            "client-a": [
                sample(
                    sample_id,
                    ce_loss,
                    source_ids=[1, 2, 3],
                    top_k_ids=[[1, 4], [2, 4], [3, 4]],
                    logit_offset=index / 10,
                )
                for index, (sample_id, ce_loss) in enumerate(
                    zip(self.sample_ids, [0.40, 0.35, 0.25, 0.10])
                )
            ],
        }
        reports = {
            client_id: self.safety_reports[client_id]
            for client_id in selected
        }

        with tempfile.TemporaryDirectory() as directory:
            service = CoordinatorService(
                data_dir=directory,
                host_gateway=HostGateway("http://host", "test-token"),
            )
            reference_bundle = SimpleNamespace(reference_samples=[object()])
            encoded = [
                SimpleNamespace(sample_id=sample_id, labels=self.labels[sample_id])
                for sample_id in self.sample_ids
            ]
            profiles = {
                self.profile.profile_id: self.profile,
                self.granite_profile.profile_id: self.granite_profile,
            }
            with (
                mock.patch(
                    "coordinator.service.resolve_alignment_profile",
                    side_effect=lambda profile_id: profiles[profile_id],
                ),
                mock.patch(
                    "coordinator.service.load_pinned_tokenizer",
                    side_effect=[
                        self.host_tokenizer,
                        self.client_tokenizer,
                        self.granite_client_tokenizer,
                    ],
                ) as tokenizer_loader,
                mock.patch.object(
                    service,
                    "_host_reference_dataset_bundle",
                    return_value=reference_bundle,
                ),
                mock.patch(
                    "coordinator.service.encode_reference_samples",
                    return_value=encoded,
                ),
            ):
                job = service._prepare_host_training_job(  # noqa: SLF001
                    manifest=manifest,
                    baseline=self.host_package,
                    baseline_samples=self.host_samples,
                    packages=mixed_packages,
                    client_samples=mixed_samples,
                    reports=reports,
                )

            self.assertEqual(tokenizer_loader.call_count, 3)
            self.assertEqual(job.accepted_client_ids, selected)
            audit = service.store.read_json(
                "rounds/round-mixed/host_training_job/integration_audit.json"
            )
            self.assertEqual(
                [value["alignment_profile_id"] for value in audit["alignments"]],
                [self.profile.profile_id, self.granite_profile.profile_id],
            )
            self.assertEqual(
                [value["client_ids"] for value in audit["alignments"]],
                [["client-b"], ["client-a"]],
            )
            self.assertTrue(
                service.store.path(
                    "rounds/round-mixed/host_training_job/"
                    "trainer_inputs.safetensors"
                ).is_file()
            )

    def test_mixed_client_profiles_use_independent_signed_mappings(self) -> None:
        mixed_packages = [
            package(
                value.sender_id,
                "client",
                self.granite_client_endpoint,
                self.sample_ids,
                self.granite_profile.profile_id,
            )
            if value.sender_id == "client-a"
            else value
            for value in self.client_packages
        ]
        mixed_samples = dict(self.client_samples)
        mixed_samples["client-a"] = [
            sample(
                sample_id,
                ce_loss,
                source_ids=[1, 2, 3],
                top_k_ids=[[1, 4], [2, 4], [3, 4]],
                logit_offset=index / 10,
            )
            for index, (sample_id, ce_loss) in enumerate(
                zip(
                    self.sample_ids,
                    [0.40, 0.35, 0.25, 0.10],
                )
            )
        ]

        with tempfile.TemporaryDirectory() as directory:
            result = self.integrate(
                directory,
                profile=None,
                client_tokenizer=None,
                alignment_profiles={
                    self.profile.profile_id: self.profile,
                    self.granite_profile.profile_id: self.granite_profile,
                },
                client_tokenizers={
                    self.profile.profile_id: self.client_tokenizer,
                    self.granite_profile.profile_id: (
                        self.granite_client_tokenizer
                    ),
                },
                client_packages=mixed_packages,
                client_samples=mixed_samples,
            )

        self.assertEqual(result.dataset.accepted_client_ids, ["client-b", "client-a"])
        self.assertEqual(
            [sample.teacher_id for sample in result.dataset.samples],
            ["host", "host", "client-b", "client-a"],
        )
        self.assertEqual(
            [value.alignment_profile_id for value in result.audit.alignments],
            [self.profile.profile_id, self.granite_profile.profile_id],
        )
        self.assertEqual(
            [value.client_ids for value in result.audit.alignments],
            [["client-b"], ["client-a"]],
        )
        self.assertEqual(result.dataset.samples[2].trust_score, 0.5)
        self.assertEqual(result.dataset.samples[3].trust_score, 1.0)

    def test_empty_aligned_rows_use_host_fallback_and_are_audited(self) -> None:
        import torch

        def empty_first_row(**values):
            logits, indices = transform_step_logits(**values)
            return [[], *logits[1:]], [[], *indices[1:]]

        with tempfile.TemporaryDirectory() as directory:
            result = self.integrate(directory, aligner=empty_first_row)

        client_winners = [
            index
            for index, value in enumerate(result.dataset.samples)
            if value.teacher_id != "host"
        ]
        self.assertEqual(client_winners, [2, 3])
        self.assertEqual(result.audit.empty_aligned_row_fallback_count, 2)
        for index in client_winners:
            torch.testing.assert_close(
                result.sparse_targets.probabilities[index, 0, :2],
                torch.softmax(torch.tensor([3.0, 0.0]), dim=-1),
            )

    def test_alignment_failure_rejects_whole_client_and_rechecks_quorum(self) -> None:
        client_samples = dict(self.client_samples)
        client_samples["client-a"] = [
            sample(
                sample_id,
                0.01,
                source_ids=[14, 15, 16],
                top_k_ids=[[14, 13], [15, 13], [16, 13]],
            )
            for sample_id in self.sample_ids
        ]

        def fail_client_a(**values):
            if values["blending_model_input_ids"][0] == 14:
                raise ValueError("synthetic alignment failure")
            return transform_step_logits(**values)

        with tempfile.TemporaryDirectory() as directory:
            result = self.integrate(
                directory,
                client_samples=client_samples,
                aligner=fail_client_a,
                trusted_client_quorum=1,
            )
            expected_mapping = VocabularyMappingCache(directory).resolve(
                profile=self.profile,
                direction="client_to_host",
                source=self.client_tokenizer,
                target=self.host_tokenizer,
                requested_token_ids={
                    token_id
                    for value in self.client_samples["client-b"]
                    for token_id in (
                        value.source_input_ids
                        + [token for row in value.top_k_token_ids for token in row]
                    )
                },
            )
            self.assertEqual(
                result.audit.alignments[0].mapping_identity_sha256,
                expected_mapping.mapping.identity_sha256,
            )
            self.assertEqual(result.dataset.accepted_client_ids, ["client-b"])
            rejection = next(
                value
                for value in result.audit.rejected_clients
                if value.client_id == "client-a"
            )
            self.assertEqual(rejection.reasons, ["alignment failed (ValueError)"])

            with self.assertRaisesRegex(
                TrustedClientQuorumError,
                "1/2",
            ):
                self.integrate(
                    directory,
                    client_samples=client_samples,
                    aligner=fail_client_a,
                )

    def test_corrupt_shared_mapping_aborts_instead_of_falling_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.integrate(directory)
            paths = sorted(Path(directory).rglob("*.json"))
            self.assertEqual(len(paths), 1)
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            payload["entries"][0]["target_token"] = "tampered"
            paths[0].write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                DistillationIntegrationError,
                "cache failed validation",
            ):
                self.integrate(directory)

    def test_answer_labels_are_required_and_score_does_not_change_targets(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                DistillationIntegrationError,
                "labels must match",
            ):
                self.integrate(
                    directory,
                    labels_by_sample={"host-best": [-100, 2, 3]},
                )

            boundary = self.integrate(directory)
            reports = dict(self.safety_reports)
            reports["client-b"] = SafetyReport(accepted=True, trust_score=1.0)
            maximum = self.integrate(directory, safety_reports=reports)
        torch.testing.assert_close(
            boundary.sparse_targets.probabilities,
            maximum.sparse_targets.probabilities,
        )
        self.assertEqual(
            [sample.teacher_id for sample in boundary.dataset.samples],
            [sample.teacher_id for sample in maximum.dataset.samples],
        )


if __name__ == "__main__":
    unittest.main()
