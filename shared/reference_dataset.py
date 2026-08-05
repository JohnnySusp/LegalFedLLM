from __future__ import annotations

from typing import Literal

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