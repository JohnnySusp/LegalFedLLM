from __future__ import annotations

import math

from shared.protocol import KnowledgePackage, KnowledgeSample, SafetyReport


MINIMUM_TRUST_SCORE = 0.5


def is_eligible_for_distillation(report: SafetyReport) -> bool:
    """Return the round-level trust gate without turning trust into a weight."""

    return report.accepted and report.trust_score >= MINIMUM_TRUST_SCORE


def inspect_knowledge_package(
    package: KnowledgePackage,
    samples: list[KnowledgeSample],
    *,
    maximum_absolute_logit: float = 100.0,
    maximum_ce_loss: float = 1_000.0,
) -> SafetyReport:
    reasons: list[str] = []
    total_values = 0
    extreme_values = 0

    if package.sample_ids != [sample.sample_id for sample in samples]:
        reasons.append("loaded sample order differs from the Knowledge Package")

    for sample in samples:
        if sample.ce_loss > maximum_ce_loss:
            reasons.append(f"sample {sample.sample_id} has an excessive CE loss")
        for row in sample.top_k_logits:
            for value in row:
                total_values += 1
                if not math.isfinite(value):
                    reasons.append(f"sample {sample.sample_id} contains non-finite logits")
                elif abs(value) > maximum_absolute_logit:
                    extreme_values += 1

    extreme_ratio = extreme_values / total_values if total_values else 1.0
    if extreme_ratio > 0.01:
        reasons.append("more than one percent of logits exceed the configured range")

    accepted = not reasons
    trust_score = 1.0 if accepted else 0.0
    return SafetyReport(accepted=accepted, trust_score=trust_score, reasons=reasons)
