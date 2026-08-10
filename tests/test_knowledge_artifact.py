from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from safetensors.numpy import load as load_safetensors
from safetensors.numpy import save as save_safetensors

from shared.crypto import sha256_hex
from shared.knowledge_artifact import (
    load_knowledge_artifact,
    ordered_sample_ids_sha256,
    serialize_knowledge_artifact,
    write_knowledge_artifact,
)
from shared.protocol import KnowledgeArtifactDescriptor, KnowledgeSample


def samples() -> list[KnowledgeSample]:
    return [
        KnowledgeSample(
            sample_id="sample-a",
            source_input_ids=[11, 12],
            attention_length=2,
            top_k_token_ids=[[101, 102], [103, 104]],
            top_k_logits=[[2.5, 1.25], [3.0, -0.5]],
            ce_loss=0.25,
        ),
        KnowledgeSample(
            sample_id="sample-b",
            source_input_ids=[21, 22, 23],
            attention_length=2,
            top_k_token_ids=[[201, 202], [203, 204], [205, 206]],
            top_k_logits=[[4.0, 2.0], [1.5, 1.0], [0.5, -1.0]],
            ce_loss=0.75,
        ),
    ]


def matching_descriptor(
    artifact: bytes,
    original: KnowledgeArtifactDescriptor,
    **changes,
) -> KnowledgeArtifactDescriptor:
    payload = original.model_dump(mode="json")
    payload.update(
        byte_size=len(artifact),
        sha256=sha256_hex(artifact),
    )
    payload.update(changes)
    return KnowledgeArtifactDescriptor.model_validate(payload)


class KnowledgeArtifactTests(unittest.TestCase):
    def test_descriptor_rejects_unknown_format_and_schema(self) -> None:
        _, descriptor = serialize_knowledge_artifact(samples())
        payload = descriptor.model_dump(mode="json")

        with self.assertRaises(ValidationError):
            KnowledgeArtifactDescriptor.model_validate(
                {**payload, "format": "npz"}
            )
        with self.assertRaises(ValidationError):
            KnowledgeArtifactDescriptor.model_validate(
                {**payload, "schema_version": "2.0"}
            )

    def test_ordered_sample_id_hash_binds_order(self) -> None:
        first = ordered_sample_ids_sha256(["sample-a", "sample-b"])
        second = ordered_sample_ids_sha256(["sample-b", "sample-a"])

        self.assertNotEqual(first, second)
        with self.assertRaises(ValueError):
            ordered_sample_ids_sha256(["sample-a", "sample-a"])

    def test_serialization_is_byte_deterministic(self) -> None:
        first_artifact, first_descriptor = serialize_knowledge_artifact(samples())
        second_artifact, second_descriptor = serialize_knowledge_artifact(samples())

        self.assertEqual(first_artifact, second_artifact)
        self.assertEqual(first_descriptor, second_descriptor)

    def test_variable_length_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.safetensors"
            original = samples()
            descriptor = write_knowledge_artifact(
                path,
                original,
                maximum_bytes=1024 * 1024,
            )
            loaded = load_knowledge_artifact(
                path,
                descriptor,
                [sample.sample_id for sample in original],
                maximum_bytes=1024 * 1024,
            )

        self.assertEqual(loaded, original)
        self.assertEqual(descriptor.sample_count, 2)
        self.assertEqual(descriptor.total_token_count, 5)
        self.assertEqual(descriptor.top_k, 2)

    def test_write_is_atomic_and_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "nested" / "knowledge.safetensors"
            descriptor = write_knowledge_artifact(
                path,
                samples(),
                maximum_bytes=1024 * 1024,
            )

            self.assertEqual(path.stat().st_size, descriptor.byte_size)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_oversized_write_and_load_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.safetensors"
            artifact, descriptor = serialize_knowledge_artifact(samples())

            with self.assertRaises(ValueError):
                write_knowledge_artifact(
                    path,
                    samples(),
                    maximum_bytes=len(artifact) - 1,
                )

            path.write_bytes(artifact)
            with self.assertRaises(ValueError):
                load_knowledge_artifact(
                    path,
                    descriptor,
                    ["sample-a", "sample-b"],
                    maximum_bytes=len(artifact) - 1,
                )

    def test_tampered_or_trailing_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.safetensors"
            artifact, descriptor = serialize_knowledge_artifact(samples())
            tampered = bytearray(artifact)
            tampered[-1] ^= 1
            path.write_bytes(tampered)

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_knowledge_artifact(
                    path,
                    descriptor,
                    ["sample-a", "sample-b"],
                    maximum_bytes=1024 * 1024,
                )

            trailing = artifact + b"x"
            path.write_bytes(trailing)
            trailing_descriptor = matching_descriptor(trailing, descriptor)
            with self.assertRaisesRegex(ValueError, "invalid safetensors"):
                load_knowledge_artifact(
                    path,
                    trailing_descriptor,
                    ["sample-a", "sample-b"],
                    maximum_bytes=1024 * 1024,
                )

            truncated = artifact[:-1]
            path.write_bytes(truncated)
            truncated_descriptor = matching_descriptor(truncated, descriptor)
            with self.assertRaisesRegex(ValueError, "invalid safetensors"):
                load_knowledge_artifact(
                    path,
                    truncated_descriptor,
                    ["sample-a", "sample-b"],
                    maximum_bytes=1024 * 1024,
                )

    def test_declared_size_and_sample_order_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.safetensors"
            descriptor = write_knowledge_artifact(
                path,
                samples(),
                maximum_bytes=1024 * 1024,
            )

            wrong_size = descriptor.model_copy(
                update={"byte_size": descriptor.byte_size + 1}
            )
            with self.assertRaisesRegex(ValueError, "byte size"):
                load_knowledge_artifact(
                    path,
                    wrong_size,
                    ["sample-a", "sample-b"],
                    maximum_bytes=1024 * 1024,
                )

            with self.assertRaisesRegex(ValueError, "ordered sample IDs"):
                load_knowledge_artifact(
                    path,
                    descriptor,
                    ["sample-b", "sample-a"],
                    maximum_bytes=1024 * 1024,
                )

    def test_missing_and_unknown_tensors_are_rejected(self) -> None:
        artifact, descriptor = serialize_knowledge_artifact(samples())
        tensors = load_safetensors(artifact)

        for changed in (
            {name: value for name, value in tensors.items() if name != "ce_losses"},
            {**tensors, "unknown": np.array([1], dtype=np.int64)},
        ):
            changed_artifact = save_safetensors(changed)
            changed_descriptor = matching_descriptor(
                changed_artifact,
                descriptor,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "knowledge.safetensors"
                path.write_bytes(changed_artifact)
                with self.assertRaisesRegex(ValueError, "tensor names"):
                    load_knowledge_artifact(
                        path,
                        changed_descriptor,
                        ["sample-a", "sample-b"],
                        maximum_bytes=1024 * 1024,
                    )

    def test_artifact_metadata_is_rejected(self) -> None:
        artifact, descriptor = serialize_knowledge_artifact(samples())
        tensors = load_safetensors(artifact)
        changed_artifact = save_safetensors(
            tensors,
            metadata={"unexpected": "value"},
        )
        changed_descriptor = matching_descriptor(changed_artifact, descriptor)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.safetensors"
            path.write_bytes(changed_artifact)
            with self.assertRaisesRegex(ValueError, "metadata"):
                load_knowledge_artifact(
                    path,
                    changed_descriptor,
                    ["sample-a", "sample-b"],
                    maximum_bytes=1024 * 1024,
                )

    def test_wrong_dtype_and_shape_are_rejected(self) -> None:
        artifact, descriptor = serialize_knowledge_artifact(samples())
        tensors = load_safetensors(artifact)
        changed_values = [
            {
                **tensors,
                "source_input_ids": tensors["source_input_ids"].astype(
                    np.int64
                ),
            },
            {
                **tensors,
                "top_k_logits": tensors["top_k_logits"][:, :1],
            },
        ]

        for changed in changed_values:
            changed_artifact = save_safetensors(changed)
            changed_descriptor = matching_descriptor(
                changed_artifact,
                descriptor,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "knowledge.safetensors"
                path.write_bytes(changed_artifact)
                with self.assertRaises(ValueError):
                    load_knowledge_artifact(
                        path,
                        changed_descriptor,
                        ["sample-a", "sample-b"],
                        maximum_bytes=1024 * 1024,
                    )

    def test_invalid_offsets_and_attention_lengths_are_rejected(self) -> None:
        artifact, descriptor = serialize_knowledge_artifact(samples())
        tensors = load_safetensors(artifact)
        changed_values = [
            {
                **tensors,
                "sample_offsets": np.array([0, 0, 5], dtype=np.int64),
            },
            {
                **tensors,
                "attention_lengths": np.array([3, 2], dtype=np.int64),
            },
        ]

        for changed in changed_values:
            changed_artifact = save_safetensors(changed)
            changed_descriptor = matching_descriptor(
                changed_artifact,
                descriptor,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "knowledge.safetensors"
                path.write_bytes(changed_artifact)
                with self.assertRaises(ValueError):
                    load_knowledge_artifact(
                        path,
                        changed_descriptor,
                        ["sample-a", "sample-b"],
                        maximum_bytes=1024 * 1024,
                    )

    def test_negative_ids_and_losses_are_rejected(self) -> None:
        artifact, descriptor = serialize_knowledge_artifact(samples())
        tensors = load_safetensors(artifact)
        changed_values = [
            {
                **tensors,
                "source_input_ids": np.array(
                    [-1, 12, 21, 22, 23], dtype=np.int32
                ),
            },
            {
                **tensors,
                "top_k_token_ids": np.array(
                    [[-1, 2]] * 5, dtype=np.int32
                ),
            },
            {
                **tensors,
                "ce_losses": np.array([-0.25, 0.75], dtype=np.float32),
            },
        ]

        for changed in changed_values:
            changed_artifact = save_safetensors(changed)
            changed_descriptor = matching_descriptor(
                changed_artifact,
                descriptor,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "knowledge.safetensors"
                path.write_bytes(changed_artifact)
                with self.assertRaises(ValueError):
                    load_knowledge_artifact(
                        path,
                        changed_descriptor,
                        ["sample-a", "sample-b"],
                        maximum_bytes=1024 * 1024,
                    )

    def test_non_finite_values_are_rejected(self) -> None:
        artifact, descriptor = serialize_knowledge_artifact(samples())
        tensors = load_safetensors(artifact)
        changed_values = [
            {
                **tensors,
                "top_k_logits": np.full((5, 2), np.nan, dtype=np.float32),
            },
            {
                **tensors,
                "ce_losses": np.array([np.inf, 0.75], dtype=np.float32),
            },
        ]

        for changed in changed_values:
            changed_artifact = save_safetensors(changed)
            changed_descriptor = matching_descriptor(
                changed_artifact,
                descriptor,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "knowledge.safetensors"
                path.write_bytes(changed_artifact)
                with self.assertRaises(ValueError):
                    load_knowledge_artifact(
                        path,
                        changed_descriptor,
                        ["sample-a", "sample-b"],
                        maximum_bytes=1024 * 1024,
                    )

    def test_float32_overflow_is_rejected_before_serialization(self) -> None:
        value = samples()[0]
        value.top_k_logits[0][0] = 1e300

        with self.assertRaisesRegex(ValueError, "finite float32"):
            serialize_knowledge_artifact([value])

    def test_invalid_source_token_ids_are_rejected_before_serialization(self) -> None:
        negative = samples()[0]
        negative.source_input_ids[0] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            serialize_knowledge_artifact([negative])

        oversized = samples()[0]
        oversized.source_input_ids[0] = 2**31
        with self.assertRaisesRegex(ValueError, "int32"):
            serialize_knowledge_artifact([oversized])


if __name__ == "__main__":
    unittest.main()
