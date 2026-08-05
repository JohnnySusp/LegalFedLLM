from __future__ import annotations

import unittest

from pydantic import ValidationError

from shared.reference_dataset import ReferenceSample


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


if __name__ == "__main__":
    unittest.main()