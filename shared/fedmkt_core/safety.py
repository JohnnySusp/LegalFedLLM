from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
import statistics

from shared.protocol import (
    ClientTrustHistory,
    KnowledgePackage,
    KnowledgeSample,
    SafetyReport,
)


MINIMUM_TRUST_SCORE = 0.15
NEUTRAL_SCORE = 0.5
HISTORY_DECAY = 0.95
TRUST_WEIGHTS = {
    "distribution_safety": 0.20,
    "loss_consistency": 0.20,
    "host_relative": 0.25,
    "peer_consistency": 0.20,
    "historical_reliability": 0.15,
}


def is_eligible_for_distillation(report: SafetyReport) -> bool:
    """Return the round-level trust gate without turning trust into a weight."""

    return (
        report.probe_stage == "post_alignment"
        and report.accepted
        and report.trust_score >= MINIMUM_TRUST_SCORE
    )


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _round_score(value: float) -> float:
    return round(_clamp01(value) + 1e-12, 2)


def _weighted_score(components: Mapping[str, float]) -> float:
    return _round_score(
        sum(TRUST_WEIGHTS[name] * components[name] for name in TRUST_WEIGHTS)
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _aggregate_risk(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    risks = [_clamp01(value) for value in values]
    return _clamp01(
        0.60 * statistics.fmean(risks)
        + 0.30 * _percentile(risks, 0.95)
        + 0.10 * max(risks)
    )


def _robust_two_sided_risks(values: Sequence[float]) -> list[float]:
    if len(values) < 3:
        return [0.0] * len(values)
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    if mad <= 1e-12:
        return [0.0] * len(values)
    scale = 1.4826 * mad
    risks: list[float] = []
    for value in values:
        z_score = abs(value - median) / scale
        risks.append(_clamp01((z_score - 3.5) / 4.5))
    return risks


def _ramp(value: float, start: float, end: float) -> float:
    if value <= start:
        return 0.0
    if value >= end:
        return 1.0
    return (value - start) / (end - start)


def _row_probability_summary(
    logits: Sequence[float],
    full_logsumexp: float,
) -> tuple[list[float], float, float, float, float]:
    probabilities = [math.exp(float(value) - full_logsumexp) for value in logits]
    top_mass = math.fsum(probabilities)
    tail_mass = max(0.0, 1.0 - top_mass)
    ordered = sorted(probabilities, reverse=True)
    top1 = ordered[0] if ordered else 0.0
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    margin = max(0.0, top1 - top2)
    buckets = [*probabilities, tail_mass]
    entropy = -math.fsum(value * math.log(value) for value in buckets if value > 0.0)
    maximum_entropy = math.log(len(buckets)) if len(buckets) > 1 else 1.0
    coarsened_entropy = entropy / maximum_entropy if maximum_entropy else 0.0
    return probabilities, tail_mass, top1, margin, _clamp01(coarsened_entropy)


def _evidence_tolerance(value: float) -> float:
    return max(2e-4, 2e-4 * max(1.0, abs(value)))


def _pre_alignment_components(
    samples: Sequence[KnowledgeSample],
) -> tuple[float, float, dict[str, float]]:
    row_records: list[tuple[str, float, float, float, float, int, int]] = []
    supervised_records: list[tuple[str, float, int, int, float]] = []

    for sample in samples:
        for row_index, (token_ids, logits, lse) in enumerate(
            zip(
                sample.top_k_token_ids,
                sample.top_k_logits,
                sample.full_logsumexp,
                strict=True,
            )
        ):
            _, tail_mass, top1, margin, entropy = _row_probability_summary(
                logits,
                lse,
            )
            top_token = token_ids[max(range(len(logits)), key=logits.__getitem__)]
            row_records.append(
                (
                    sample.sample_id,
                    top1,
                    margin,
                    tail_mass,
                    entropy,
                    row_index,
                    top_token,
                )
            )
            gold_id = sample.gold_token_ids[row_index]
            if gold_id != -100:
                supervised_records.append(
                    (
                        sample.sample_id,
                        sample.gold_token_nll[row_index],
                        gold_id,
                        top_token,
                        top1,
                    )
                )

    top1_values = [record[1] for record in row_records]
    margin_values = [record[2] for record in row_records]
    tail_values = [record[3] for record in row_records]
    entropy_values = [record[4] for record in row_records]
    top1_outliers = _robust_two_sided_risks(top1_values)
    margin_outliers = _robust_two_sided_risks(margin_values)
    tail_outliers = _robust_two_sided_risks(tail_values)
    entropy_outliers = _robust_two_sided_risks(entropy_values)

    distribution_by_sample: dict[str, list[float]] = {
        sample.sample_id: [] for sample in samples
    }
    for index, record in enumerate(row_records):
        sample_id, top1, margin, tail, entropy, _row_index, _top_token = record
        over_sharp = _ramp(top1, 0.98, 0.9999)
        coarsened_flat = min(
            _ramp(entropy, 0.98, 0.9999),
            _ramp(0.02 - margin, 0.0, 0.02),
        )
        tail_dominated_flat = min(
            _ramp(tail, 0.98, 0.9999),
            _ramp(0.02 - top1, 0.0, 0.02),
        )
        over_flat = max(coarsened_flat, tail_dominated_flat)
        robust = max(
            top1_outliers[index],
            margin_outliers[index],
            tail_outliers[index],
            entropy_outliers[index],
        )
        distribution_by_sample[sample_id].append(max(over_sharp, over_flat, robust))

    nll_values = [record[1] for record in supervised_records]
    nll_outliers = _robust_two_sided_risks(nll_values)
    loss_by_sample: dict[str, list[float]] = {sample.sample_id: [] for sample in samples}
    for index, record in enumerate(supervised_records):
        sample_id, _nll, gold_id, top_token, top1 = record
        high_confidence_wrong = 0.0
        if top_token != gold_id:
            high_confidence_wrong = _ramp(top1, 0.80, 0.995)
        loss_by_sample[sample_id].append(
            max(high_confidence_wrong, nll_outliers[index])
        )

    pattern_risk = 0.0
    if len(row_records) >= 8:
        top_tokens = [record[6] for record in row_records]
        dominant_fraction = max(Counter(top_tokens).values()) / len(top_tokens)
        summaries = [
            (
                round(record[1], 6),
                round(record[2], 6),
                round(record[3], 6),
                round(record[4], 6),
                record[6],
            )
            for record in row_records
        ]
        repeated_fraction = max(Counter(summaries).values()) / len(summaries)
        pattern_risk = max(
            _ramp(dominant_fraction, 0.95, 1.0),
            _ramp(repeated_fraction, 0.95, 1.0),
        )

    sample_risks: dict[str, float] = {}
    distribution_risks: list[float] = []
    loss_risks: list[float] = []
    for sample in samples:
        distribution_risk = max(
            _aggregate_risk(distribution_by_sample[sample.sample_id]),
            pattern_risk,
        )
        loss_risk = _aggregate_risk(loss_by_sample[sample.sample_id])
        distribution_risks.append(distribution_risk)
        loss_risks.append(loss_risk)
        sample_risks[sample.sample_id] = _clamp01(
            0.5 * distribution_risk + 0.5 * loss_risk
        )

    distribution_safety = 1.0 - _aggregate_risk(distribution_risks)
    # Internally consistent Client-reported loss evidence is neutral, not
    # affirmative trust. It can only penalize the package until reliability is
    # earned from cross-round evidence.
    loss_consistency = 0.5 * (1.0 - _aggregate_risk(loss_risks))
    return (
        _clamp01(distribution_safety),
        _clamp01(loss_consistency),
        sample_risks,
    )


def inspect_knowledge_package(
    package: KnowledgePackage,
    samples: list[KnowledgeSample],
    *,
    maximum_absolute_logit: float = 100.0,
    maximum_ce_loss: float = 1_000.0,
) -> SafetyReport:
    """Run hard evidence checks and the tokenizer-independent soft probe.

    The pre-alignment score is provisional. A package that passes hard checks is
    allowed to reach token alignment even when this provisional score is below
    the final trust threshold; the final eligibility decision is made only after
    Host-relative and peer evidence can be calculated.
    """

    reasons: list[str] = []
    total_values = 0
    extreme_values = 0

    if package.sample_ids != [sample.sample_id for sample in samples]:
        reasons.append("loaded sample order differs from the Knowledge Package")

    for sample in samples:
        if sample.ce_loss > maximum_ce_loss:
            reasons.append(f"sample {sample.sample_id} has an excessive CE loss")
        supervised_nll: list[float] = []
        for row_index, (token_ids, logits, lse, gold_id, gold_logit, reported_nll) in enumerate(
            zip(
                sample.top_k_token_ids,
                sample.top_k_logits,
                sample.full_logsumexp,
                sample.gold_token_ids,
                sample.gold_token_logits,
                sample.gold_token_nll,
                strict=True,
            )
        ):
            if len(token_ids) != len(set(token_ids)):
                reasons.append(
                    f"sample {sample.sample_id} row {row_index} repeats a top-k token ID"
                )
            maximum_logit = max(logits)
            if lse + _evidence_tolerance(lse) < maximum_logit:
                reasons.append(
                    f"sample {sample.sample_id} row {row_index} has an invalid full logsumexp"
                )
            probabilities = [math.exp(float(value) - lse) for value in logits]
            if math.fsum(probabilities) > 1.0 + 1e-4:
                reasons.append(
                    f"sample {sample.sample_id} row {row_index} has top-k probability mass above one"
                )
            if gold_id != -100:
                recomputed_nll = lse - gold_logit
                if recomputed_nll < -_evidence_tolerance(recomputed_nll):
                    reasons.append(
                        f"sample {sample.sample_id} row {row_index} has negative recomputed gold NLL"
                    )
                if abs(recomputed_nll - reported_nll) > _evidence_tolerance(recomputed_nll):
                    reasons.append(
                        f"sample {sample.sample_id} row {row_index} has inconsistent gold NLL evidence"
                    )
                if gold_id in token_ids:
                    transmitted_gold_logit = logits[token_ids.index(gold_id)]
                    if abs(transmitted_gold_logit - gold_logit) > _evidence_tolerance(gold_logit):
                        reasons.append(
                            f"sample {sample.sample_id} row {row_index} has inconsistent gold-token logit"
                        )
                elif gold_logit > min(logits) + _evidence_tolerance(gold_logit):
                    reasons.append(
                        f"sample {sample.sample_id} row {row_index} omits a gold logit that belongs in top-k"
                    )
                supervised_nll.append(max(0.0, recomputed_nll))
            for value in logits:
                total_values += 1
                if abs(value) > maximum_absolute_logit:
                    extreme_values += 1
        if supervised_nll:
            recomputed_ce = statistics.fmean(supervised_nll)
            if abs(recomputed_ce - sample.ce_loss) > _evidence_tolerance(recomputed_ce):
                reasons.append(
                    f"sample {sample.sample_id} has CE inconsistent with gold-token evidence"
                )

    extreme_ratio = extreme_values / total_values if total_values else 1.0
    if extreme_ratio > 0.01:
        reasons.append("more than one percent of logits exceed the configured range")

    if reasons:
        return SafetyReport(
            accepted=False,
            trust_score=0.0,
            reasons=list(dict.fromkeys(reasons)),
            probe_stage="pre_alignment",
            score_components={},
            sample_risks={},
        )

    distribution_safety, loss_consistency, sample_risks = _pre_alignment_components(
        samples
    )
    components = {
        "distribution_safety": distribution_safety,
        "loss_consistency": loss_consistency,
        "host_relative": NEUTRAL_SCORE,
        "peer_consistency": NEUTRAL_SCORE,
        "historical_reliability": NEUTRAL_SCORE,
    }
    return SafetyReport(
        accepted=True,
        trust_score=_weighted_score(components),
        reasons=[],
        probe_stage="pre_alignment",
        score_components=components,
        sample_risks={key: _clamp01(value) for key, value in sample_risks.items()},
    )


def _sparse_softmax(token_ids: Sequence[int], logits: Sequence[float]) -> dict[int, float]:
    if not token_ids or not logits:
        return {}
    maximum = max(float(value) for value in logits)
    exponentials = [math.exp(float(value) - maximum) for value in logits]
    denominator = math.fsum(exponentials)
    if denominator <= 0.0:
        return {}
    result: dict[int, float] = {}
    for token_id, value in zip(token_ids, exponentials, strict=True):
        result[token_id] = result.get(token_id, 0.0) + value / denominator
    return result


def _js_divergence(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    if not left or not right:
        return 1.0
    keys = set(left) | set(right)
    divergence = 0.0
    for key in keys:
        p = left.get(key, 0.0)
        q = right.get(key, 0.0)
        midpoint = 0.5 * (p + q)
        if p > 0.0:
            divergence += 0.5 * p * math.log(p / midpoint)
        if q > 0.0:
            divergence += 0.5 * q * math.log(q / midpoint)
    return _clamp01(divergence / math.log(2.0))


def _sample_distribution_divergence(
    host_sample: KnowledgeSample,
    aligned_ids: Sequence[Sequence[int]],
    aligned_logits: Sequence[Sequence[float]],
) -> float:
    row_risks: list[float] = []
    for row_index, gold_id in enumerate(host_sample.gold_token_ids):
        if gold_id == -100:
            continue
        host_distribution = _sparse_softmax(
            host_sample.top_k_token_ids[row_index],
            host_sample.top_k_logits[row_index],
        )
        client_distribution = _sparse_softmax(
            aligned_ids[row_index],
            aligned_logits[row_index],
        )
        row_risks.append(_js_divergence(host_distribution, client_distribution))
    return _aggregate_risk(row_risks)


def _quality_conditioned_risk(
    divergence: float,
    source_ce: float,
    reference_ce: float,
) -> float:
    if source_ce <= reference_ce:
        return 0.0
    quality_gap = source_ce - reference_ce
    return _clamp01(divergence * (1.0 - math.exp(-quality_gap)))


def finalize_aligned_safety_reports(
    *,
    host_samples: Sequence[KnowledgeSample],
    client_samples: Mapping[str, Sequence[KnowledgeSample]],
    aligned_by_client: Mapping[
        str,
        Mapping[str, tuple[Sequence[Sequence[int]], Sequence[Sequence[float]], int]],
    ],
    pre_alignment_reports: Mapping[str, SafetyReport],
    selected_client_ids: Sequence[str],
    historical_reliability: Mapping[str, float] | None = None,
) -> dict[str, SafetyReport]:
    """Finalize package trust after outputs share the Host token space."""

    host_by_id = {sample.sample_id: sample for sample in host_samples}
    histories = historical_reliability or {}
    active_ids = [
        client_id
        for client_id in selected_client_ids
        if client_id in aligned_by_client
        and client_id in client_samples
        and pre_alignment_reports.get(client_id) is not None
        and pre_alignment_reports[client_id].accepted
    ]
    client_by_id = {
        client_id: {sample.sample_id: sample for sample in client_samples[client_id]}
        for client_id in active_ids
    }

    host_risks: dict[str, dict[str, float]] = {client_id: {} for client_id in active_ids}
    for client_id in active_ids:
        for sample_id, host_sample in host_by_id.items():
            client_sample = client_by_id[client_id][sample_id]
            aligned_ids, aligned_logits, _fallbacks = aligned_by_client[client_id][sample_id]
            divergence = _sample_distribution_divergence(
                host_sample,
                aligned_ids,
                aligned_logits,
            )
            host_risks[client_id][sample_id] = _quality_conditioned_risk(
                divergence,
                client_sample.ce_loss,
                host_sample.ce_loss,
            )

    peer_risks: dict[str, dict[str, float]] = {
        client_id: {sample_id: 0.0 for sample_id in host_by_id}
        for client_id in active_ids
    }
    peer_scores = {client_id: NEUTRAL_SCORE for client_id in active_ids}
    if len(active_ids) >= 3:
        for client_id in active_ids:
            for sample_id, host_sample in host_by_id.items():
                client_sample = client_by_id[client_id][sample_id]
                aligned_ids, aligned_logits, _ = aligned_by_client[client_id][sample_id]
                source_rows = [
                    _sparse_softmax(ids, logits)
                    for ids, logits in zip(aligned_ids, aligned_logits, strict=True)
                ]
                peer_values: list[float] = []
                peer_ces: list[float] = []
                for peer_id in active_ids:
                    if peer_id == client_id:
                        continue
                    peer_ids, peer_logits, _ = aligned_by_client[peer_id][sample_id]
                    row_values: list[float] = []
                    for row_index, gold_id in enumerate(host_sample.gold_token_ids):
                        if gold_id == -100:
                            continue
                        peer_distribution = _sparse_softmax(
                            peer_ids[row_index],
                            peer_logits[row_index],
                        )
                        row_values.append(
                            _js_divergence(source_rows[row_index], peer_distribution)
                        )
                    peer_values.append(_aggregate_risk(row_values))
                    peer_ces.append(client_by_id[peer_id][sample_id].ce_loss)
                peer_divergence = statistics.median(peer_values) if peer_values else 0.0
                peer_reference_ce = statistics.median(peer_ces) if peer_ces else client_sample.ce_loss
                peer_risks[client_id][sample_id] = _quality_conditioned_risk(
                    peer_divergence,
                    client_sample.ce_loss,
                    peer_reference_ce,
                )
            peer_scores[client_id] = 0.5 * (
                1.0 - _aggregate_risk(list(peer_risks[client_id].values()))
            )

    finalized: dict[str, SafetyReport] = {}
    for client_id in active_ids:
        pre = pre_alignment_reports[client_id]
        distribution_safety = pre.score_components.get(
            "distribution_safety", NEUTRAL_SCORE
        )
        loss_consistency = pre.score_components.get(
            "loss_consistency", NEUTRAL_SCORE
        )
        host_component = 0.5 * (
            1.0 - _aggregate_risk(list(host_risks[client_id].values()))
        )
        history_component = _clamp01(histories.get(client_id, NEUTRAL_SCORE))
        components = {
            "distribution_safety": distribution_safety,
            "loss_consistency": loss_consistency,
            "host_relative": _clamp01(host_component),
            "peer_consistency": _clamp01(peer_scores[client_id]),
            "historical_reliability": history_component,
        }
        trust_score = _weighted_score(components)
        accepted = trust_score >= MINIMUM_TRUST_SCORE
        reasons = [] if accepted else [
            f"post-alignment trust score {trust_score:.2f} is below {MINIMUM_TRUST_SCORE:.2f}"
        ]
        current_risks: dict[str, float] = {}
        for sample_id in host_by_id:
            pre_risk = pre.sample_risks.get(sample_id, 0.0)
            current_risks[sample_id] = _clamp01(
                (0.40 * pre_risk + 0.25 * host_risks[client_id][sample_id]
                 + 0.20 * peer_risks[client_id][sample_id])
                / 0.85
            )
        finalized[client_id] = SafetyReport(
            accepted=accepted,
            trust_score=trust_score,
            reasons=reasons,
            probe_stage="post_alignment",
            score_components=components,
            sample_risks=current_risks,
        )
    return finalized


def new_client_trust_history(client_id: str) -> ClientTrustHistory:
    return ClientTrustHistory(
        client_id=client_id,
        alpha=1.0,
        beta=1.0,
        completed_rounds=0,
        last_round_id=None,
    )


def update_client_trust_history(
    history: ClientTrustHistory,
    *,
    round_id: str,
    sample_risks: Mapping[str, float],
    decay: float = HISTORY_DECAY,
) -> ClientTrustHistory:
    if history.last_round_id == round_id:
        return history
    if not 0.0 < decay <= 1.0:
        raise ValueError("history decay must be in (0, 1]")
    risks = [_clamp01(value) for value in sample_risks.values()]
    alpha = decay * history.alpha + math.fsum(1.0 - value for value in risks)
    beta = decay * history.beta + math.fsum(risks)
    return ClientTrustHistory(
        client_id=history.client_id,
        alpha=alpha,
        beta=beta,
        completed_rounds=history.completed_rounds + 1,
        last_round_id=round_id,
    )
