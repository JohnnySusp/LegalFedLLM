from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

try:
    from scripts.validate_fedmkt_alignment import run_validation
    from shared.reference_dataset import (
        ReferenceSample,
        reference_dataset_identity,
        write_reference_jsonl,
    )
    from shared.tokenizer_validation import ValidatedTokenizer
except ModuleNotFoundError as exc:
    VALIDATION_IMPORT_ERROR = exc
else:
    VALIDATION_IMPORT_ERROR = None


class FakeTokenizer:
    is_fast = True

    def __init__(self, marker: str) -> None:
        self.vocabulary = {
            f"{marker}token-{index}": index for index in range(32)
        }
        self.tokens = {
            token_id: token for token, token_id in self.vocabulary.items()
        }

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocabulary)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str | None]:
        return [self.tokens.get(token_id) for token_id in token_ids]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_options,
    ) -> list[int]:
        if not tokenize:
            raise AssertionError("the validation runner must request token IDs")
        values = [1, 2]
        if messages[-1]["role"] == "assistant":
            values.extend([3, 4])
        elif not add_generation_prompt:
            raise AssertionError("user-only prompts require a generation marker")
        return values


def reference_samples(count: int) -> list[ReferenceSample]:
    return [
        ReferenceSample(
            dataset_id="validation-fixture",
            dataset_version="1",
            sample_id=f"sample-{index:03d}",
            chapter="Chapter",
            section="Section",
            question=f"Question {index}?",
            gold_answer=f"Answer {index}.",
        )
        for index in range(count)
    ]


@unittest.skipUnless(
    TORCH_AVAILABLE and VALIDATION_IMPORT_ERROR is None,
    f"optional ML dependencies are unavailable: {VALIDATION_IMPORT_ERROR}",
)
class FedMKTValidationRunnerTests(unittest.TestCase):
    def test_mixed_client_report_is_deterministic_and_machine_readable(self) -> None:
        samples = reference_samples(4)
        identity = reference_dataset_identity(samples)

        def load_fake(endpoint, **_options):
            return ValidatedTokenizer(
                endpoint=endpoint,
                tokenizer=FakeTokenizer(endpoint.word_boundary_marker),
                artifact_sha256=endpoint.tokenizer_artifact_sha256,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.jsonl"
            output_path = root / "report.json"
            write_reference_jsonl(reference_path, samples)
            report = run_validation(
                reference_path=reference_path,
                output_path=output_path,
                mapping_cache_dir=root / "mapping-cache",
                identity_dir=root / "identities",
                expected_sample_count=identity.sample_count,
                expected_dataset_hash=identity.dataset_hash,
                tokenizer_loader=load_fake,
            )
            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report, loaded)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(
            report["result"]["accepted_client_ids"],
            ["client-b", "client-a"],
        )
        self.assertEqual(
            report["result"]["selected_source_counts"],
            {"host": 2, "client-b": 1, "client-a": 1},
        )
        self.assertTrue(report["determinism"]["identical"])
        self.assertEqual(
            report["tokenization"],
            {
                "maximum_sequence_length": 4096,
                "overlength_policy": "reject_without_truncation",
                "qwen_client": {
                    "sample_count": 4,
                    "total_token_count": 16,
                    "maximum_observed_tokens": 4,
                    "maximum_observed_sample_ids": [
                        "sample-000",
                        "sample-001",
                        "sample-002",
                        "sample-003",
                    ],
                },
                "granite_client_and_validation_host": {
                    "sample_count": 4,
                    "total_token_count": 16,
                    "maximum_observed_tokens": 4,
                    "maximum_observed_sample_ids": [
                        "sample-000",
                        "sample-001",
                        "sample-002",
                        "sample-003",
                    ],
                },
            },
        )
        self.assertEqual(
            report["knowledge_artifact_policy"],
            {
                "top_k": 4,
                "maximum_logical_package_bytes": 25 * 1024 * 1024,
                "position_retention": "all_non_padding_source_positions",
                "logit_source": "deterministic_validation_fixture",
            },
        )
        self.assertTrue(
            all(value["top_k"] == 4 for value in report["packages"].values())
        )
        self.assertEqual(report["resources"]["vram"]["status"], "not_applicable")
        self.assertIsNone(report["resources"]["vram"]["peak_bytes"])
        self.assertEqual(
            [
                value["alignment_profile_id"]
                for value in report["alignment_mappings"]
            ],
            [
                "dtw:qwen3-1.7b--granite3.3-2b-v1",
                "dtw:granite3.3-2b-client--granite3.3-2b-host-v1",
            ],
        )

    def test_wrong_frozen_dataset_identity_fails_before_tokenizer_loading(self) -> None:
        samples = reference_samples(1)

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("tokenizer loader should not be called")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.jsonl"
            write_reference_jsonl(reference_path, samples)
            with self.assertRaisesRegex(ValueError, "expected 565"):
                run_validation(
                    reference_path=reference_path,
                    output_path=root / "report.json",
                    mapping_cache_dir=root / "mapping-cache",
                    identity_dir=root / "identities",
                    tokenizer_loader=fail_if_called,
                )


if __name__ == "__main__":
    unittest.main()
