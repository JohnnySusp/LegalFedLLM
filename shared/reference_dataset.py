from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


REFERENCE_DATASET_SCHEMA_VERSION = 1


class ReferenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ReferenceSource(ReferenceModel):
    document_id: str = Field(min_length=1, max_length=256)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_pages(self) -> "ReferenceSource":
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must be provided together")

        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end cannot precede page_start")

        return self


class ReferenceSample(ReferenceModel):
    schema_version: Literal[1] = REFERENCE_DATASET_SCHEMA_VERSION
    dataset_id: str = Field(min_length=1, max_length=256)
    dataset_version: str = Field(min_length=1, max_length=256)
    sample_id: str = Field(min_length=1, max_length=256)
    chapter: str = Field(min_length=1, max_length=1024)
    section: str = Field(min_length=1, max_length=2048)
    question: str = Field(min_length=1, max_length=100_000)
    gold_answer: str = Field(min_length=1, max_length=1_000_000)
    source: ReferenceSource | None = None

    @field_validator(
        "dataset_id",
        "dataset_version",
        "sample_id",
        "chapter",
        "section",
        "question",
        "gold_answer",
    )
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text fields must not be blank")
        return value


def validate_reference_samples(
    samples: Iterable[ReferenceSample],
) -> list[ReferenceSample]:
    values = list(samples)

    if not values:
        raise ValueError("reference dataset must contain at least one sample")

    dataset_id = values[0].dataset_id
    dataset_version = values[0].dataset_version
    sample_ids: set[str] = set()

    for sample in values:
        if sample.dataset_id != dataset_id:
            raise ValueError("reference dataset contains multiple dataset IDs")

        if sample.dataset_version != dataset_version:
            raise ValueError("reference dataset contains multiple dataset versions")

        if sample.sample_id in sample_ids:
            raise ValueError(f"duplicate sample ID: {sample.sample_id}")

        sample_ids.add(sample.sample_id)

    return values


def write_reference_jsonl(
    path: str | Path,
    samples: Iterable[ReferenceSample],
) -> None:
    values = validate_reference_samples(samples)
    target = Path(path)

    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in values:
            payload = sample.model_dump(mode="json")
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def load_reference_jsonl(
    path: str | Path,
) -> list[ReferenceSample]:
    source = Path(path)
    samples: list[ReferenceSample] = []

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"blank line in reference dataset at line {line_number}"
                )

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in reference dataset at line {line_number}"
                ) from exc

            try:
                sample = ReferenceSample.model_validate(payload)
            except ValueError as exc:
                raise ValueError(
                    f"invalid reference sample at line {line_number}"
                ) from exc

            samples.append(sample)

    return validate_reference_samples(samples)


def write_pretty_reference_json(
    jsonl_path: str | Path,
    pretty_path: str | Path,
) -> None:
    samples = load_reference_jsonl(jsonl_path)
    payload = [sample.model_dump(mode="json") for sample in samples]

    Path(pretty_path).write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )