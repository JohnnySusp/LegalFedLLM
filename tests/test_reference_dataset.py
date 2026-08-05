from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from shared.reference_dataset import (
    ReferenceSample,
    ReferenceSource,
    load_reference_jsonl,
    reference_dataset_hash,
    reference_dataset_hash_payload,
    reference_dataset_identity,
    split_reference_samples,
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


def section_samples(
    *,
    chapter_number: int,
    section_number: int,
    count: int,
    chapter: str,
    section: str,
) -> list[ReferenceSample]:
    return [
        sample(
            sample_id=(
                f"example-ch{chapter_number:03d}"
                f"-s{section_number:03d}"
                f"-q{question_number:03d}"
            ),
            chapter=chapter,
            section=section,
            question=f"What is example question {question_number}?",
            gold_answer=f"This is example answer {question_number}.",
        )
        for question_number in range(1, count + 1)
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

    def test_same_dataset_has_same_hash(self) -> None:
        first = samples()
        second = samples()

        self.assertEqual(
            reference_dataset_hash(first),
            reference_dataset_hash(second),
        )

    def test_dataset_hash_is_sha256_hex(self) -> None:
        value = reference_dataset_hash(samples())

        self.assertEqual(len(value), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in value))

    def test_gold_answer_changes_dataset_hash(self) -> None:
        original = samples()
        changed = samples()

        changed[0].gold_answer = "A different gold answer."

        self.assertNotEqual(
            reference_dataset_hash(original),
            reference_dataset_hash(changed),
        )

    def test_chapter_changes_dataset_hash(self) -> None:
        original = samples()
        changed = samples()

        changed[0].chapter = "Different Chapter"

        self.assertNotEqual(
            reference_dataset_hash(original),
            reference_dataset_hash(changed),
        )

    def test_sample_order_changes_dataset_hash(self) -> None:
        original = samples()
        reordered = samples()

        reordered[0], reordered[1] = reordered[1], reordered[0]

        self.assertNotEqual(
            reference_dataset_hash(original),
            reference_dataset_hash(reordered),
        )

    def test_source_pages_do_not_change_dataset_hash(self) -> None:
        original = samples()
        changed = samples()

        source = changed[0].source
        assert source is not None

        changed[0].source = ReferenceSource(
            document_id=source.document_id,
            page_start=100,
            page_end=101,
        )

        self.assertEqual(
            reference_dataset_hash(original),
            reference_dataset_hash(changed),
        )

    def test_dataset_identity(self) -> None:
        values = samples()

        identity = reference_dataset_identity(values)

        self.assertEqual(identity.schema_version, 1)
        self.assertEqual(identity.dataset_id, "example-reference")
        self.assertEqual(identity.dataset_version, "v1")
        self.assertEqual(identity.sample_count, 3)
        self.assertEqual(
            identity.dataset_hash,
            reference_dataset_hash(values),
        )

    def test_hash_payload_excludes_source_provenance(self) -> None:
        payload = reference_dataset_hash_payload(samples())

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["dataset_id"], "example-reference")
        self.assertEqual(payload["dataset_version"], "v1")
        self.assertNotIn("source", payload["samples"][0])
        self.assertEqual(
            list(payload["samples"][0]),
            [
                "sample_id",
                "chapter",
                "section",
                "question",
                "gold_answer",
            ],
        )

    def test_jsonl_round_trip_preserves_dataset_hash(self) -> None:
        values = samples()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"

            before = reference_dataset_hash(values)

            write_reference_jsonl(path, values)
            loaded = load_reference_jsonl(path)

            after = reference_dataset_hash(loaded)

        self.assertEqual(before, after)

    def test_split_uses_floor_per_section(self) -> None:
        values = section_samples(
            chapter_number=1,
            section_number=1,
            count=20,
            chapter="Example Chapter",
            section="Example Section",
        )

        reference, validation = split_reference_samples(values)

        self.assertEqual(len(reference), 16)
        self.assertEqual(len(validation), 4)

        self.assertEqual(
            [value.sample_id for value in reference],
            [
                f"example-ch001-s001-q{number:03d}"
                for number in range(1, 17)
            ],
        )
        self.assertEqual(
            [value.sample_id for value in validation],
            [
                f"example-ch001-s001-q{number:03d}"
                for number in range(17, 21)
            ],
        )

    def test_singleton_section_is_reference_only(self) -> None:
        values = section_samples(
            chapter_number=1,
            section_number=1,
            count=1,
            chapter="Example Chapter",
            section="Singleton Section",
        )

        reference, validation = split_reference_samples(values)

        self.assertEqual(
            [value.sample_id for value in reference],
            ["example-ch001-s001-q001"],
        )
        self.assertEqual(validation, [])

    def test_each_section_is_split_independently(self) -> None:
        first_section = section_samples(
            chapter_number=1,
            section_number=1,
            count=3,
            chapter="Example Chapter",
            section="First Section",
        )
        second_section = section_samples(
            chapter_number=1,
            section_number=2,
            count=2,
            chapter="Example Chapter",
            section="Second Section",
        )

        reference, validation = split_reference_samples(
            first_section + second_section
        )

        self.assertEqual(
            [value.sample_id for value in reference],
            [
                "example-ch001-s001-q001",
                "example-ch001-s001-q002",
                "example-ch001-s002-q001",
            ],
        )
        self.assertEqual(
            [value.sample_id for value in validation],
            [
                "example-ch001-s001-q003",
                "example-ch001-s002-q002",
            ],
        )

    def test_same_section_name_in_different_chapters_is_split_separately(
        self,
    ) -> None:
        first_chapter = section_samples(
            chapter_number=1,
            section_number=1,
            count=2,
            chapter="First Chapter",
            section="Shared Section Name",
        )
        second_chapter = section_samples(
            chapter_number=2,
            section_number=1,
            count=2,
            chapter="Second Chapter",
            section="Shared Section Name",
        )

        reference, validation = split_reference_samples(
            first_chapter + second_chapter
        )

        self.assertEqual(
            [value.sample_id for value in reference],
            [
                "example-ch001-s001-q001",
                "example-ch002-s001-q001",
            ],
        )
        self.assertEqual(
            [value.sample_id for value in validation],
            [
                "example-ch001-s001-q002",
                "example-ch002-s001-q002",
            ],
        )

    def test_split_is_deterministic(self) -> None:
        values = section_samples(
            chapter_number=1,
            section_number=1,
            count=7,
            chapter="Example Chapter",
            section="Example Section",
        )

        first_reference, first_validation = split_reference_samples(values)
        second_reference, second_validation = split_reference_samples(values)

        self.assertEqual(
            [value.sample_id for value in first_reference],
            [value.sample_id for value in second_reference],
        )
        self.assertEqual(
            [value.sample_id for value in first_validation],
            [value.sample_id for value in second_validation],
        )

if __name__ == "__main__":
    unittest.main()