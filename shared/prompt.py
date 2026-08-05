from __future__ import annotations

from shared.reference_dataset import ReferenceSample


PROMPT_TEMPLATE_ID = "chapter-section-question-v1"

PROMPT_TEMPLATE = (
    "Chapter: {chapter}\n\n"
    "Section: {section}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def render_reference_prompt(sample: ReferenceSample) -> str:
    return PROMPT_TEMPLATE.format(
        chapter=sample.chapter,
        section=sample.section,
        question=sample.question,
    )