from __future__ import annotations

from collections.abc import Mapping

from shared.protocol import (
    KnowledgePackage,
    SafetyReport,
    ValidatedDistillationDataset,
    ValidatedDistillationSample,
)


def dual_min_ce_select(
    *,
    host_package: KnowledgePackage,
    client_packages: list[KnowledgePackage],
    safety_reports: Mapping[str, SafetyReport],
) -> ValidatedDistillationDataset:
    host_samples = {sample.sample_id: sample for sample in host_package.samples}
    client_samples = {
        package.sender_id: {sample.sample_id: sample for sample in package.samples}
        for package in client_packages
    }
    selected: list[ValidatedDistillationSample] = []

    for sample_id in host_package.sample_ids:
        host_sample = host_samples[sample_id]
        candidates: list[tuple[float, str, float]] = []
        for package in client_packages:
            report = safety_reports[package.sender_id]
            if not report.accepted or report.trust_score <= 0:
                continue
            candidates.append(
                (
                    client_samples[package.sender_id][sample_id].ce_loss,
                    package.sender_id,
                    report.trust_score,
                )
            )

        if not candidates:
            continue
        teacher_loss, teacher_id, trust_weight = min(candidates, key=lambda item: item[0])
        if teacher_loss >= host_sample.ce_loss:
            continue
        teacher = client_samples[teacher_id][sample_id]
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
                trust_weight=trust_weight,
            )
        )

    return ValidatedDistillationDataset.create(
        round_id=host_package.round_id,
        manifest_hash=host_package.manifest_hash,
        host_adapter_version=host_package.adapter_version,
        accepted_client_ids=sorted(package.sender_id for package in client_packages),
        samples=selected,
    )
