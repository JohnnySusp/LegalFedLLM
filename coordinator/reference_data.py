from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared.reference_dataset import (
    ReferenceDatasetIdentity,
    ReferenceSample,
    load_reference_jsonl,
    reference_dataset_identity,
    write_reference_jsonl,
)


@dataclass(frozen=True)
class CoordinatorReferenceData:
    reference_samples: tuple[ReferenceSample, ...]
    validation_samples: tuple[ReferenceSample, ...]
    reference_identity: ReferenceDatasetIdentity
    validation_identity: ReferenceDatasetIdentity

    @classmethod
    def load(
        cls,
        reference_path: str | Path,
        validation_path: str | Path,
    ) -> "CoordinatorReferenceData":
        reference_samples = load_reference_jsonl(reference_path)
        validation_samples = load_reference_jsonl(validation_path)

        reference_identity = reference_dataset_identity(reference_samples)
        validation_identity = reference_dataset_identity(validation_samples)

        if reference_identity.dataset_id != validation_identity.dataset_id:
            raise ValueError(
                "reference and validation datasets have different dataset IDs"
            )

        if (
            reference_identity.dataset_version
            != validation_identity.dataset_version
        ):
            raise ValueError(
                "reference and validation datasets have different versions"
            )

        reference_ids = {
            sample.sample_id
            for sample in reference_samples
        }
        validation_ids = {
            sample.sample_id
            for sample in validation_samples
        }
        overlap = reference_ids & validation_ids

        if overlap:
            sample_id = sorted(overlap)[0]
            raise ValueError(
                f"sample appears in both reference and validation data: "
                f"{sample_id}"
            )

        return cls(
            reference_samples=tuple(reference_samples),
            validation_samples=tuple(validation_samples),
            reference_identity=reference_identity,
            validation_identity=validation_identity,
        )

    @property
    def sample_ids(self) -> list[str]:
        return [
            sample.sample_id
            for sample in self.reference_samples
        ]

    def write_snapshot(
        self,
        reference_path: str | Path,
        validation_path: str | Path,
    ) -> None:
        reference_target = Path(reference_path)
        validation_target = Path(validation_path)

        reference_target.parent.mkdir(parents=True, exist_ok=True)
        validation_target.parent.mkdir(parents=True, exist_ok=True)

        write_reference_jsonl(
            reference_target,
            self.reference_samples,
        )
        write_reference_jsonl(
            validation_target,
            self.validation_samples,
        )