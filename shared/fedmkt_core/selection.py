from __future__ import annotations

from collections.abc import Mapping

from shared.protocol import (
    KnowledgePackage,
    KnowledgeSample,
    SafetyReport,
    ValidatedDistillationDataset,
    ValidatedDistillationSample,
)
from shared.fedmkt_core.safety import is_eligible_for_distillation


def dual_min_ce_select(
    *,
    host_package: KnowledgePackage,
    host_samples: list[KnowledgeSample],
    client_packages: list[KnowledgePackage],
    client_samples: Mapping[str, list[KnowledgeSample]],
    safety_reports: Mapping[str, SafetyReport],
    selected_client_ids: list[str],
) -> ValidatedDistillationDataset:
    host_by_id = {sample.sample_id: sample for sample in host_samples}
    client_by_id = {
        sender_id: {sample.sample_id: sample for sample in samples}
        for sender_id, samples in client_samples.items()
    }
    packages_by_id = {package.sender_id: package for package in client_packages}
    selected: list[ValidatedDistillationSample] = []

    for sample_id in host_package.sample_ids:
        host_sample = host_by_id[sample_id]
        candidates = [
            (host_sample.ce_loss, host_package.sender_id, 1.0, host_sample)
        ]
        for client_id in selected_client_ids:
            if client_id not in packages_by_id:
                continue
            report = safety_reports[client_id]
            if not is_eligible_for_distillation(report):
                continue
            candidates.append(
                (
                    client_by_id[client_id][sample_id].ce_loss,
                    client_id,
                    report.trust_score,
                    client_by_id[client_id][sample_id],
                )
            )

        teacher_loss, teacher_id, trust_score, teacher = min(
            candidates,
            key=lambda item: item[0],
        )
        selected.append(
            ValidatedDistillationSample(
                sample_id=sample_id,
                teacher_id=teacher_id,
                teacher_ce_loss=teacher_loss,
                host_ce_loss=host_sample.ce_loss,
                source_input_ids=host_sample.source_input_ids,
                attention_length=host_sample.attention_length,
                aligned_top_k_token_ids=teacher.top_k_token_ids,
                aligned_top_k_logits=teacher.top_k_logits,
                trust_score=trust_score,
            )
        )

    return ValidatedDistillationDataset.create(
        round_id=host_package.round_id,
        manifest_hash=host_package.manifest_hash,
        host_adapter_version=host_package.adapter_version,
        accepted_client_ids=[
            client_id
            for client_id in selected_client_ids
            if client_id in packages_by_id
            and is_eligible_for_distillation(safety_reports[client_id])
        ],
        samples=selected,
    )
