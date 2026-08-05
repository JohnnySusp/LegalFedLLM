from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from shared.reference_dataset import (
    ReferenceSample,
    load_reference_jsonl,
    write_pretty_reference_json,
    write_reference_jsonl,
)


def sample(**changes) -> ReferenceSample:
    payload = {
        "schema_version": 1,
        "dataset_id": "example-reference",
        "dataset_version": "v1",
        "sample_id": "example-ch001-s001-q001",
        "chapter": "Example Chapter",
        "section": "Example Section",
        "question": "What is the example question?",
        "gold_answer": "This is the example answer.",
        "source": {
            "document_id": "example-document",
            "page_start": 10,
            "page_end": 11,
        },
    }
    payload.update(changes)
    return ReferenceSample.model_validate(payload)


def samples() -> list[ReferenceSample]:
    return [
        sample(
            sample_id="example-ch001-s001-q001",
            question="What is the first example question?",
            gold_answer="This is the first example answer.",
        ),
        sample(
            sample_id="example-ch001-s001-q002",
            question="What is the second example question?",
            gold_answer="This is the second example answer.",
        ),
        sample(
            sample_id="example-ch001-s001-q003",
            question="What is the third example question?",
            gold_answer="This is the third example answer.",
        ),
    ]


class ReferenceSampleTests(unittest.TestCase):
    def test_valid_sample(self) -> None:
        value = sample()

        self.assertEqual(value.schema_version, 1)
        self.assertEqual(value.dataset_id, "example-reference")
        self.assertEqual(value.sample_id, "example-ch001-s001-q001")

    def test_blank_gold_answer_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            sample(gold_answer="   ")

    def test_invalid_source_page_range_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            sample(
                source={
                    "document_id": "example-document",
                    "page_start": 11,
                    "page_end": 10,
                }
            )

    def test_unknown_schema_version_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            sample(schema_version=2)

    def test_jsonl_round_trip_preserves_order(self) -> None:
        values = samples()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"

            write_reference_jsonl(path, values)
            loaded = load_reference_jsonl(path)

        self.assertEqual(
            [value.sample_id for value in loaded],
            [
                "example-ch001-s001-q001",
                "example-ch001-s001-q002",
                "example-ch001-s001-q003",
            ],
        )

    def test_jsonl_round_trip_preserves_text(self) -> None:
        value = sample(
            question="A question with Greek text: Ελλάδα;",
            gold_answer="First paragraph.\n\nSecond paragraph.",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"

            write_reference_jsonl(path, [value])
            loaded = load_reference_jsonl(path)

        self.assertEqual(loaded[0].question, value.question)
        self.assertEqual(loaded[0].gold_answer, value.gold_answer)

    def test_duplicate_sample_id_is_rejected(self) -> None:
        duplicate = [
            sample(sample_id="example-ch001-s001-q001"),
            sample(sample_id="example-ch001-s001-q001"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"

            with self.assertRaises(ValueError):
                write_reference_jsonl(path, duplicate)

    def test_mixed_dataset_versions_are_rejected(self) -> None:
        mixed = [
            sample(sample_id="example-ch001-s001-q001"),
            sample(
                sample_id="example-ch001-s001-q002",
                dataset_version="v2",
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"

            with self.assertRaises(ValueError):
                write_reference_jsonl(path, mixed)

    def test_blank_jsonl_line_is_rejected(self) -> None:
        value = sample()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"

            write_reference_jsonl(path, [value])

            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n")

            with self.assertRaises(ValueError):
                load_reference_jsonl(path)

    def test_pretty_json_is_generated_from_jsonl(self) -> None:
        values = samples()

        with tempfile.TemporaryDirectory() as directory:
            jsonl_path = Path(directory) / "reference.jsonl"
            pretty_path = Path(directory) / "reference.pretty.json"

            write_reference_jsonl(jsonl_path, values)
            write_pretty_reference_json(jsonl_path, pretty_path)

            payload = json.loads(
                pretty_path.read_text(encoding="utf-8")
            )

        self.assertEqual(len(payload), 3)
        self.assertEqual(
            payload[0]["sample_id"],
            "example-ch001-s001-q001",
        )
        self.assertEqual(
            payload[2]["sample_id"],
            "example-ch001-s001-q003",
        )


if __name__ == "__main__":
    unittest.main()