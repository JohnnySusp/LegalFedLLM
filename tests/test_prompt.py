from __future__ import annotations

import unittest

from shared.prompt import (
    PROMPT_TEMPLATE_ID,
    render_reference_prompt,
)
from shared.reference_dataset import ReferenceSample


class PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = ReferenceSample.model_validate(
            {
                "schema_version": 1,
                "dataset_id": "example-reference",
                "dataset_version": "v1",
                "sample_id": "example-ch001-s001-q001",
                "chapter": "Example Chapter",
                "section": "Example Section",
                "question": "What is the example question?",
                "gold_answer": "This is the example answer.",
            }
        )

    def test_prompt_template_id(self) -> None:
        self.assertEqual(
            PROMPT_TEMPLATE_ID,
            "chapter-section-question-v1",
        )

    def test_render_reference_prompt(self) -> None:
        prompt = render_reference_prompt(self.sample)

        self.assertEqual(
            prompt,
            "Chapter: Example Chapter\n\n"
            "Section: Example Section\n\n"
            "Question: What is the example question?\n\n"
            "Answer:",
        )

    def test_gold_answer_is_not_in_prompt(self) -> None:
        prompt = render_reference_prompt(self.sample)

        self.assertNotIn(self.sample.gold_answer, prompt)


if __name__ == "__main__":
    unittest.main()