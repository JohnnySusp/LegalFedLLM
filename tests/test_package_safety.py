from __future__ import annotations

import math
import tempfile
import unittest

from shared.fedmkt_core.safety import (
    MINIMUM_TRUST_SCORE,
    finalize_aligned_safety_reports,
    inspect_knowledge_package,
    is_eligible_for_distillation,
    new_client_trust_history,
    update_client_trust_history,
)
from coordinator.service import CoordinatorService
from shared.protocol import ClientTrustHistory, KnowledgePackage, KnowledgeSample, SafetyReport


def package(*sample_ids: str) -> KnowledgePackage:
    return KnowledgePackage.model_construct(sample_ids=list(sample_ids))


def sample(
    sample_id: str,
    *,
    logits: list[list[float]],
    token_ids: list[list[int]] | None = None,
    gold_token_id: int | None = None,
    ce_loss: float = 0.5,
    logsumexp: list[float] | None = None,
) -> KnowledgeSample:
    length = len(logits)
    ids = token_ids or [list(range(10, 10 + len(row))) for row in logits]
    if logsumexp is None:
        logsumexp = [
            math.log(sum(math.exp(value) for value in row) + math.exp(max(row) - 1.0))
            for row in logits
        ]
    gold = gold_token_id if gold_token_id is not None else ids[0][0]
    gold_logit = logsumexp[0] - ce_loss
    return KnowledgeSample(
        sample_id=sample_id,
        source_input_ids=list(range(1, length + 1)),
        attention_length=length,
        top_k_token_ids=ids,
        top_k_logits=logits,
        full_logsumexp=logsumexp,
        gold_token_ids=[gold, *([-100] * (length - 1))],
        gold_token_logits=[gold_logit, *([0.0] * (length - 1))],
        gold_token_nll=[ce_loss, *([0.0] * (length - 1))],
        ce_loss=ce_loss,
    )


class PackageSafetyTests(unittest.TestCase):
    def test_pre_alignment_recomputes_ce_from_gold_evidence(self) -> None:
        value = sample("s1", logits=[[2.0, 0.0], [1.0, 0.0]])
        payload = value.model_dump(mode="python")
        payload["ce_loss"] = 0.1
        forged = KnowledgeSample.model_validate(payload)

        report = inspect_knowledge_package(package("s1"), [forged])

        self.assertFalse(report.accepted)
        self.assertEqual(report.trust_score, 0.0)
        self.assertEqual(report.probe_stage, "pre_alignment")
        self.assertTrue(any("CE inconsistent" in reason for reason in report.reasons))

    def test_pre_alignment_score_is_two_decimal_and_not_a_final_gate(self) -> None:
        normal = sample(
            "s1",
            logits=[[2.0, 0.0], [1.5, 0.0]],
            logsumexp=[2.5, 2.1],
            ce_loss=0.5,
        )
        report = inspect_knowledge_package(package("s1"), [normal])

        self.assertTrue(report.accepted)
        self.assertEqual(report.probe_stage, "pre_alignment")
        self.assertEqual(report.trust_score, round(report.trust_score, 2))
        self.assertFalse(is_eligible_for_distillation(report))
        self.assertEqual(report.score_components["host_relative"], 0.5)
        self.assertEqual(report.score_components["peer_consistency"], 0.5)

    def test_excessively_sharp_and_flat_rows_reduce_distribution_safety(self) -> None:
        normal = sample(
            "normal",
            logits=[[2.0, 0.0]],
            logsumexp=[2.5],
            ce_loss=0.5,
        )
        sharp = sample(
            "sharp",
            logits=[[10.0, -10.0]],
            logsumexp=[10.0005],
            ce_loss=0.0005,
        )
        flat = sample(
            "flat",
            logits=[[0.0, 0.0]],
            logsumexp=[math.log(3.0)],
            ce_loss=math.log(3.0),
        )

        normal_score = inspect_knowledge_package(
            package("normal"), [normal]
        ).score_components["distribution_safety"]
        sharp_score = inspect_knowledge_package(
            package("sharp"), [sharp]
        ).score_components["distribution_safety"]
        flat_score = inspect_knowledge_package(
            package("flat"), [flat]
        ).score_components["distribution_safety"]

        self.assertLess(sharp_score, normal_score)
        self.assertLess(flat_score, normal_score)

    def test_single_client_peer_evidence_is_neutral(self) -> None:
        host = sample("s1", logits=[[2.0, 0.0]], logsumexp=[2.5], ce_loss=0.5)
        client = sample("s1", logits=[[2.0, 0.0]], logsumexp=[2.4], ce_loss=0.4)
        pre = inspect_knowledge_package(package("s1"), [client])

        reports = finalize_aligned_safety_reports(
            host_samples=[host],
            client_samples={"client-a": [client]},
            aligned_by_client={
                "client-a": {
                    "s1": (client.top_k_token_ids, client.top_k_logits, 0)
                }
            },
            pre_alignment_reports={"client-a": pre},
            selected_client_ids=["client-a"],
        )

        report = reports["client-a"]
        self.assertEqual(report.probe_stage, "post_alignment")
        self.assertEqual(report.score_components["peer_consistency"], 0.5)
        self.assertTrue(is_eligible_for_distillation(report))

    def test_host_disagreement_penalizes_only_when_client_quality_is_worse(self) -> None:
        host = sample(
            "s1",
            logits=[[5.0, 0.0]],
            token_ids=[[10, 11]],
            logsumexp=[5.5],
            ce_loss=0.5,
        )
        worse = sample(
            "s1",
            logits=[[5.0, 0.0]],
            token_ids=[[20, 21]],
            logsumexp=[7.0],
            ce_loss=2.0,
        )
        better = sample(
            "s1",
            logits=[[5.0, 0.0]],
            token_ids=[[20, 21]],
            logsumexp=[5.2],
            ce_loss=0.2,
        )
        pre_worse = inspect_knowledge_package(package("s1"), [worse])
        pre_better = inspect_knowledge_package(package("s1"), [better])
        aligned = ([[20, 21]], [[5.0, 0.0]], 0)

        worse_report = finalize_aligned_safety_reports(
            host_samples=[host],
            client_samples={"c": [worse]},
            aligned_by_client={"c": {"s1": aligned}},
            pre_alignment_reports={"c": pre_worse},
            selected_client_ids=["c"],
        )["c"]
        better_report = finalize_aligned_safety_reports(
            host_samples=[host],
            client_samples={"c": [better]},
            aligned_by_client={"c": {"s1": aligned}},
            pre_alignment_reports={"c": pre_better},
            selected_client_ids=["c"],
        )["c"]

        self.assertLess(
            worse_report.score_components["host_relative"],
            better_report.score_components["host_relative"],
        )
        self.assertEqual(better_report.score_components["host_relative"], 0.5)

    def test_final_threshold_controls_eligibility(self) -> None:
        host = sample("s1", logits=[[2.0, 0.0]], logsumexp=[2.5], ce_loss=0.5)
        client = sample("s1", logits=[[0.0, 2.0]], logsumexp=[4.0], ce_loss=2.0)
        pre = SafetyReport(
            accepted=True,
            trust_score=0.2,
            probe_stage="pre_alignment",
            score_components={
                "distribution_safety": 0.0,
                "loss_consistency": 0.0,
                "host_relative": 0.5,
                "peer_consistency": 0.5,
                "historical_reliability": 0.5,
            },
            sample_risks={"s1": 1.0},
        )
        report = finalize_aligned_safety_reports(
            host_samples=[host],
            client_samples={"c": [client]},
            aligned_by_client={
                "c": {"s1": (client.top_k_token_ids, client.top_k_logits, 0)}
            },
            pre_alignment_reports={"c": pre},
            selected_client_ids=["c"],
            historical_reliability={"c": 0.0},
        )["c"]

        self.assertLess(report.trust_score, MINIMUM_TRUST_SCORE)
        self.assertFalse(report.accepted)
        self.assertFalse(is_eligible_for_distillation(report))

    def test_coordinator_persists_final_report_and_history_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = CoordinatorService(
                data_dir=root,
                host_gateway=object(),
            )
            report = SafetyReport(
                accepted=True,
                trust_score=0.60,
                probe_stage="post_alignment",
                score_components={
                    "distribution_safety": 1.0,
                    "loss_consistency": 0.5,
                    "host_relative": 0.5,
                    "peer_consistency": 0.5,
                    "historical_reliability": 0.5,
                },
                sample_risks={"s1": 0.25},
            )
            service._persist_final_safety_reports(
                "round-1", {"client-a": report}
            )
            service._persist_final_safety_reports(
                "round-1", {"client-a": report}
            )

            stored_report = SafetyReport.model_validate(
                service.store.read_json("rounds/round-1/safety/client-a.json")
            )
            history = ClientTrustHistory.model_validate(
                service.store.read_json("trust_history/client-a.json")
            )
            self.assertEqual(stored_report, report)
            self.assertEqual(history.completed_rounds, 1)
            self.assertEqual(history.last_round_id, "round-1")

    def test_history_starts_neutral_decays_and_is_idempotent_per_round(self) -> None:
        history = new_client_trust_history("client-a")
        self.assertEqual(history.reliability, 0.5)

        updated = update_client_trust_history(
            history,
            round_id="round-1",
            sample_risks={"s1": 0.0, "s2": 1.0},
        )
        self.assertEqual(updated.completed_rounds, 1)
        self.assertEqual(updated.last_round_id, "round-1")
        self.assertAlmostEqual(updated.alpha, 1.95)
        self.assertAlmostEqual(updated.beta, 1.95)
        self.assertEqual(
            update_client_trust_history(
                updated,
                round_id="round-1",
                sample_risks={"s1": 1.0},
            ),
            updated,
        )


if __name__ == "__main__":
    unittest.main()
