from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shared.client_reverse_artifact import (
    load_client_reverse_training_artifact,
    write_client_reverse_training_artifact,
)
from shared.crypto import Ed25519Identity, sha256_hex
from shared.fedmkt_core.reverse_integration import integrate_reverse_distillation
from shared.knowledge_artifact import serialize_knowledge_artifact
from shared.protocol import KnowledgePackage, KnowledgeSample, LoraProfile, ModelProfile


def profile(role: str) -> ModelProfile:
    return ModelProfile(
        profile_id=f"mock-{role}",
        role=role,
        model_id=f"legalfedllm/mock-{role}",
        model_revision="v1",
        tokenizer_id="legalfedllm/mock-tokenizer",
        tokenizer_revision="v1",
        tokenizer_class="MockTokenizer",
        training_backend="mock",
        serving_backend="mock",
        prompt_template_id="mock",
        prompt_template_hash=sha256_hex(b"mock"),
        lora=LoraProfile(rank=8),
    )


def samples(losses: tuple[float, float], offset: int) -> list[KnowledgeSample]:
    return [
        KnowledgeSample(
            sample_id=sample_id,
            source_input_ids=[1, 2, 3],
            attention_length=3,
            top_k_token_ids=[
                [4 + offset, 5 + offset],
                [6 + offset, 7 + offset],
                [8 + offset, 9 + offset],
            ],
            top_k_logits=[[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]],
            ce_loss=loss,
        )
        for sample_id, loss in zip(("s1", "s2"), losses)
    ]


def package(
    identity: Ed25519Identity,
    role: str,
    values: list[KnowledgeSample],
) -> KnowledgePackage:
    _, descriptor = serialize_knowledge_artifact(values)
    return KnowledgePackage.create_signed(
        identity=identity,
        round_id="round-1",
        manifest_hash="a" * 64,
        sender_id=role,
        sender_role=role,
        model_profile=profile(role),
        adapter_version=1,
        alignment_profile_id="mock_identity:1",
        reference_dataset_id="reference-v1",
        reference_dataset_hash="b" * 64,
        top_k=2,
        sample_ids=[value.sample_id for value in values],
        artifact=descriptor,
    )


class ReverseIntegrationTests(unittest.TestCase):
    def test_all_rows_are_retained_and_ties_use_client_self_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            identity = Ed25519Identity.load_or_create(Path(directory) / "key.pem")
            host_samples = samples((0.1, 0.5), 0)
            client_samples = samples((0.2, 0.5), 10)
            host = package(identity, "host", host_samples)
            client = package(identity, "client", client_samples)

            batch = integrate_reverse_distillation(
                client_id="client",
                parent_adapter_version=1,
                parent_adapter_hash="c" * 64,
                host_adapter_promoted=False,
                partition_hash="d" * 64,
                transfer_sample_ids=["s1", "s2"],
                host_package=host,
                host_samples=host_samples,
                client_package=client,
                client_samples=client_samples,
                labels_by_sample={"s1": [-100, -100, 3], "s2": [-100, -100, 3]},
            )

            self.assertEqual(batch.sample_ids, ("s1", "s2"))
            self.assertEqual(batch.audit.host_teacher_sample_ids, ["s1"])
            self.assertEqual(
                [sample.teacher_role for sample in batch.audit.samples],
                ["host", "client"],
            )
            self.assertFalse(batch.audit.host_adapter_promoted)

            path = Path(directory) / "reverse.safetensors"
            descriptor = write_client_reverse_training_artifact(
                path,
                batch,
                pad_token_id=0,
                maximum_bytes=1024 * 1024,
            )
            loaded = load_client_reverse_training_artifact(
                path,
                descriptor,
                batch.sample_ids,
                maximum_bytes=1024 * 1024,
                vocabulary_size=32,
            )
            self.assertEqual(loaded["input_ids"].shape, (2, 3))

    def test_real_path_uses_client_owned_host_to_client_mapping(self) -> None:
        from shared.alignment_profiles import BidirectionalAlignmentProfile
        from shared.tokenizer_validation import ValidatedTokenizer
        from shared.vocabulary_mapping import VocabularyMappingCache
        from tests.test_fedmkt_integration import FakeTokenizer, endpoint, model_profile

        host_endpoint = endpoint(
            role="host",
            profile_id="host-test",
            marker="▁",
            artifact_hash="1" * 64,
            vocabulary_size=5,
            tokenizer_vocabulary_size=5,
            pad_token_id=0,
        )
        client_endpoint = endpoint(
            role="client",
            profile_id="client-test",
            marker="Ġ",
            artifact_hash="2" * 64,
            vocabulary_size=15,
            tokenizer_vocabulary_size=15,
            pad_token_id=0,
        )
        alignment = BidirectionalAlignmentProfile(
            profile_id="dtw:test-v1",
            strategy="dtw",
            profile_version="test-v1",
            client=client_endpoint,
            host=host_endpoint,
            client_to_host_owner="coordinator",
            host_to_client_owner="client",
        )
        host_tokenizer = ValidatedTokenizer(
            endpoint=host_endpoint,
            tokenizer=FakeTokenizer({"<pad>": 0, "▁a": 1, "▁b": 2, "▁c": 3, "▁d": 4}),
            artifact_sha256=host_endpoint.tokenizer_artifact_sha256,
        )
        client_tokenizer = ValidatedTokenizer(
            endpoint=client_endpoint,
            tokenizer=FakeTokenizer(
                {
                    **{f"unused-{index}": index for index in range(10)},
                    "Ġa": 10,
                    "Ġb": 11,
                    "Ġc": 12,
                    "Ġd": 13,
                    "<pad>": 14,
                }
            ),
            artifact_sha256=client_endpoint.tokenizer_artifact_sha256,
        )
        host_sample = KnowledgeSample(
            sample_id="s1",
            source_input_ids=[1, 2, 3],
            attention_length=3,
            top_k_token_ids=[[1, 4], [2, 4], [3, 4]],
            top_k_logits=[[3.0, 0.0]] * 3,
            ce_loss=0.1,
        )
        client_sample = KnowledgeSample(
            sample_id="s1",
            source_input_ids=[10, 11, 12],
            attention_length=3,
            top_k_token_ids=[[10, 13], [11, 13], [12, 13]],
            top_k_logits=[[2.0, 0.0]] * 3,
            ce_loss=0.2,
        )
        common = {
            "round_id": "round-1",
            "manifest_hash": "a" * 64,
            "adapter_version": 1,
            "alignment_profile_id": alignment.profile_id,
            "reference_dataset_id": "reference-v1",
            "reference_dataset_hash": "b" * 64,
            "sample_ids": ["s1"],
            "top_k": 2,
        }
        host_package = KnowledgePackage.model_construct(
            **common,
            sender_id="host",
            sender_role="host",
            model_profile=model_profile(host_endpoint),
            package_hash="c" * 64,
        )
        client_package = KnowledgePackage.model_construct(
            **common,
            sender_id="client",
            sender_role="client",
            model_profile=model_profile(client_endpoint),
            package_hash="d" * 64,
        )

        with tempfile.TemporaryDirectory() as directory:
            batch = integrate_reverse_distillation(
                client_id="client",
                parent_adapter_version=1,
                parent_adapter_hash="e" * 64,
                host_adapter_promoted=True,
                partition_hash="f" * 64,
                transfer_sample_ids=["s1"],
                host_package=host_package,
                host_samples=[host_sample],
                client_package=client_package,
                client_samples=[client_sample],
                labels_by_sample={"s1": [-100, 11, 12]},
                profile=alignment,
                host_tokenizer=host_tokenizer,
                client_tokenizer=client_tokenizer,
                mapping_cache=VocabularyMappingCache(directory),
            )

        self.assertEqual(batch.audit.host_teacher_sample_ids, ["s1"])
        self.assertEqual(batch.audit.alignment.alignment_direction, "host_to_client")
        self.assertEqual(batch.sparse_targets.token_ids[0, 0, 0].item(), 10)


if __name__ == "__main__":
    unittest.main()
