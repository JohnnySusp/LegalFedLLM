from __future__ import annotations

import unittest

from shared.fedmkt_core.selection import dual_min_ce_select
from shared.protocol import KnowledgePackage, KnowledgeSample, SafetyReport


HASH = "0" * 64


def package(sender_id: str, sample_ids: list[str]) -> KnowledgePackage:
    return KnowledgePackage.model_construct(
        round_id="round-1",
        manifest_hash=HASH,
        sender_id=sender_id,
        adapter_version=0,
        sample_ids=sample_ids,
    )


def sample(sample_id: str, ce_loss: float, token_id: int) -> KnowledgeSample:
    return KnowledgeSample(
        sample_id=sample_id,
        source_input_ids=[token_id],
        attention_length=1,
        top_k_token_ids=[[token_id]],
        top_k_logits=[[float(token_id)]],
        ce_loss=ce_loss,
    )


class FedMKTSelectionTests(unittest.TestCase):
    def test_host_wins_ties_and_every_sample_is_retained(self) -> None:
        sample_ids = ["host-best", "host-tie", "client-tie", "client-best"]
        host = package("host", sample_ids)
        client_a = package("client-a", sample_ids)
        client_b = package("client-b", sample_ids)

        host_samples = [
            sample("host-best", 0.10, 10),
            sample("host-tie", 0.20, 20),
            sample("client-tie", 0.50, 30),
            sample("client-best", 0.50, 40),
        ]
        samples_by_client = {
            "client-a": [
                sample("host-best", 0.30, 110),
                sample("host-tie", 0.20, 120),
                sample("client-tie", 0.25, 130),
                sample("client-best", 0.30, 140),
            ],
            "client-b": [
                sample("host-best", 0.40, 210),
                sample("host-tie", 0.35, 220),
                sample("client-tie", 0.25, 230),
                sample("client-best", 0.15, 240),
            ],
        }
        reports = {
            client_id: SafetyReport(accepted=True, trust_score=1.0)
            for client_id in samples_by_client
        }

        dataset = dual_min_ce_select(
            host_package=host,
            host_samples=host_samples,
            client_packages=[client_a, client_b],
            client_samples=samples_by_client,
            safety_reports=reports,
            selected_client_ids=["client-b", "client-a"],
        )

        self.assertEqual([value.sample_id for value in dataset.samples], sample_ids)
        self.assertEqual(dataset.accepted_client_ids, ["client-b", "client-a"])
        selected = {value.sample_id: value for value in dataset.samples}
        self.assertEqual(selected["host-best"].teacher_id, "host")
        self.assertEqual(selected["host-tie"].teacher_id, "host")
        self.assertEqual(selected["client-tie"].teacher_id, "client-b")
        self.assertEqual(selected["client-best"].teacher_id, "client-b")
        self.assertEqual(selected["host-best"].aligned_top_k_token_ids, [[10]])
        self.assertEqual(selected["client-tie"].aligned_top_k_token_ids, [[230]])

    def test_untrusted_clients_are_excluded_and_host_sample_is_retained(self) -> None:
        host = package("host", ["sample-1"])
        client = package("client-a", ["sample-1"])

        dataset = dual_min_ce_select(
            host_package=host,
            host_samples=[sample("sample-1", 0.50, 10)],
            client_packages=[client],
            client_samples={"client-a": [sample("sample-1", 0.10, 20)]},
            safety_reports={
                "client-a": SafetyReport(
                    accepted=False,
                    trust_score=0.0,
                    reasons=["rejected"],
                )
            },
            selected_client_ids=["client-a"],
        )

        self.assertEqual(len(dataset.samples), 1)
        self.assertEqual(dataset.samples[0].teacher_id, "host")
        self.assertEqual(dataset.samples[0].trust_score, 1.0)

    def test_eligibility_requires_hard_acceptance_and_threshold_score(self) -> None:
        host = package("host", ["sample-1"])
        packages = [
            package("below", ["sample-1"]),
            package("boundary", ["sample-1"]),
            package("hard-rejected", ["sample-1"]),
        ]
        dataset = dual_min_ce_select(
            host_package=host,
            host_samples=[sample("sample-1", 0.50, 10)],
            client_packages=packages,
            client_samples={
                "below": [sample("sample-1", 0.01, 20)],
                "boundary": [sample("sample-1", 0.10, 30)],
                "hard-rejected": [sample("sample-1", 0.05, 40)],
            },
            safety_reports={
                "below": SafetyReport(accepted=True, trust_score=0.49),
                "boundary": SafetyReport(accepted=True, trust_score=0.50),
                "hard-rejected": SafetyReport(
                    accepted=False,
                    trust_score=0.99,
                    reasons=["hard check failed"],
                ),
            },
            selected_client_ids=["below", "boundary", "hard-rejected"],
        )

        self.assertEqual(dataset.accepted_client_ids, ["boundary"])
        self.assertEqual(dataset.samples[0].teacher_id, "boundary")
        self.assertEqual(dataset.samples[0].trust_score, 0.5)


if __name__ == "__main__":
    unittest.main()
