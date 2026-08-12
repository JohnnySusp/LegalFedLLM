from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from client.knowledge import (
    encode_reference_samples,
    knowledge_sample_from_rows,
)
from client.model_profiles import QWEN_PROFILE_ID, pinned_client_profile
from shared.reference_dataset import ReferenceSample


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.calls: list[tuple[list[dict[str, str]], bool, dict[str, object]]] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        **options,
    ):
        self.calls.append((messages, add_generation_prompt, options))
        prompt = messages[0]["content"]
        prefix = [11, len(prompt), 12]
        if add_generation_prompt:
            return prefix
        answer = messages[1]["content"]
        return prefix + [21, len(answer), 22]


def reference_sample(sample_id: str, answer: str = "Gold answer") -> ReferenceSample:
    return ReferenceSample(
        schema_version=1,
        dataset_id="reference-v1",
        dataset_version="v1",
        sample_id=sample_id,
        chapter="Chapter A",
        section="Section B",
        question=f"Question for {sample_id}?",
        gold_answer=answer,
    )


class ReferenceKnowledgeEncodingTests(unittest.TestCase):
    def test_reference_encoding_reuses_answer_only_chat_contract(self) -> None:
        tokenizer = FakeTokenizer()
        values = [reference_sample("sample-1"), reference_sample("sample-2")]
        encoded = encode_reference_samples(
            values,
            tokenizer=tokenizer,
            model_profile=pinned_client_profile(QWEN_PROFILE_ID),
            maximum_sequence_length=32,
            expected_sample_ids=["sample-1", "sample-2"],
        )

        self.assertEqual([item.sample_id for item in encoded], ["sample-1", "sample-2"])
        self.assertEqual(encoded[0].labels[:3], [-100, -100, -100])
        self.assertEqual(encoded[0].labels[3:], [21, 11, 22])
        prompt_call, full_call = tokenizer.calls[:2]
        self.assertEqual(
            prompt_call[0][0]["content"],
            "Chapter: Chapter A\n\nSection: Section B\n\n"
            "Question: Question for sample-1?\n\nAnswer:",
        )
        self.assertTrue(prompt_call[1])
        self.assertFalse(full_call[1])
        self.assertEqual(
            [call[2] for call in tokenizer.calls],
            [{"enable_thinking": False}] * 4,
        )

    def test_reference_order_and_all_overlength_samples_are_reported(self) -> None:
        values = [reference_sample("sample-1"), reference_sample("sample-2")]
        with self.assertRaisesRegex(ValueError, "signed D\\^P order"):
            encode_reference_samples(
                values,
                tokenizer=FakeTokenizer(),
                model_profile=pinned_client_profile(QWEN_PROFILE_ID),
                maximum_sequence_length=32,
                expected_sample_ids=["sample-2", "sample-1"],
            )

        with self.assertRaisesRegex(
            ValueError,
            r"2 reference sample\(s\).*maximum observed length=6.*sample-1=6.*sample-2=6",
        ):
            encode_reference_samples(
                values,
                tokenizer=FakeTokenizer(),
                model_profile=pinned_client_profile(QWEN_PROFILE_ID),
                maximum_sequence_length=5,
                expected_sample_ids=["sample-1", "sample-2"],
            )

    def test_padding_rows_are_removed_before_knowledge_storage(self) -> None:
        encoded = encode_reference_samples(
            [reference_sample("sample-1")],
            tokenizer=FakeTokenizer(),
            model_profile=pinned_client_profile(QWEN_PROFILE_ID),
            maximum_sequence_length=32,
            expected_sample_ids=["sample-1"],
        )[0]
        rows = [[index, index + 100] for index in range(9)]
        logits = [[float(index), -float(index)] for index in range(9)]
        sample = knowledge_sample_from_rows(
            encoded,
            top_k_token_ids=rows,
            top_k_logits=logits,
            ce_loss=0.25,
        )

        self.assertEqual(sample.attention_length, 6)
        self.assertEqual(len(sample.source_input_ids), 6)
        self.assertEqual(sample.top_k_token_ids, rows[:6])
        self.assertEqual(sample.top_k_logits, logits[:6])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for FedMKT metric tests")
class FedMKTAnswerOnlyMetricTests(unittest.TestCase):
    def test_ce_is_meaned_over_supervised_answer_tokens(self) -> None:
        import torch
        import torch.nn.functional as functional

        from shared.fedmkt_core.ml.logit_generation import Metric

        torch.manual_seed(7)
        logits = torch.randn(1, 6, 5, dtype=torch.float16)
        labels = torch.tensor([[-100, -100, -100, 1, 2, 3]])
        attention = torch.ones_like(labels)
        actual = Metric.cal_ce(logits, labels, attention, labels, SimpleNamespace())

        token_losses = functional.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, 5),
            labels[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(1, -1)
        supervised = labels[:, 1:].ne(-100)
        expected = (token_losses * supervised).sum(-1) / supervised.sum(-1)
        diluted = (token_losses * supervised).sum(-1) / attention[:, 1:].sum(-1)

        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, diluted))
        self.assertEqual(actual.dtype, torch.float32)

    def test_ce_is_invariant_to_right_padding(self) -> None:
        import torch

        from shared.fedmkt_core.ml.logit_generation import Metric

        torch.manual_seed(11)
        logits = torch.randn(1, 5, 7)
        labels = torch.tensor([[-100, -100, 2, 3, 4]])
        attention = torch.ones_like(labels)
        baseline = Metric.cal_ce(logits, labels, attention, labels, SimpleNamespace())

        padded_logits = torch.cat((logits, torch.randn(1, 3, 7)), dim=1)
        padded_labels = torch.cat((labels, torch.full((1, 3), -100)), dim=1)
        padded_attention = torch.cat(
            (attention, torch.zeros((1, 3), dtype=torch.long)), dim=1
        )
        padded = Metric.cal_ce(
            padded_logits,
            padded_labels,
            padded_attention,
            padded_labels,
            SimpleNamespace(),
        )
        torch.testing.assert_close(padded, baseline)

    def test_whole_sequence_labels_retain_fedmkt_token_mean(self) -> None:
        import torch
        import torch.nn.functional as functional

        from shared.fedmkt_core.ml.logit_generation import Metric

        torch.manual_seed(13)
        logits = torch.randn(1, 4, 6)
        labels = torch.tensor([[0, 1, 2, 3]])
        attention = torch.ones_like(labels)
        actual = Metric.cal_ce(logits, labels, attention, labels, SimpleNamespace())
        expected = functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, 6),
            labels[:, 1:].reshape(-1),
            reduction="mean",
        ).reshape(1)
        torch.testing.assert_close(actual, expected)

    def test_ce_rejects_a_sample_without_supervised_targets(self) -> None:
        import torch

        from shared.fedmkt_core.ml.logit_generation import Metric

        with self.assertRaisesRegex(ValueError, "supervised target token"):
            Metric.cal_ce(
                torch.zeros(1, 3, 4),
                torch.zeros(1, 3, dtype=torch.long),
                torch.ones(1, 3, dtype=torch.long),
                torch.full((1, 3), -100, dtype=torch.long),
                SimpleNamespace(),
            )

    def test_generation_uses_eval_no_grad_no_cache_and_raw_top_k(self) -> None:
        import torch

        from shared.fedmkt_core.ml.logit_generation import generate_pub_data_logits
        from shared.fedmkt_core.ml.vars_define import PER_STEP_INDICES, PER_STEP_LOGITS

        observations: dict[str, object] = {}

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, input_ids, attention_mask, use_cache):
                observations["training"] = self.training
                observations["grad_enabled"] = torch.is_grad_enabled()
                observations["use_cache"] = use_cache
                values = torch.arange(6, dtype=torch.float16).view(1, 1, 6)
                return SimpleNamespace(
                    logits=values.repeat(input_ids.size(0), input_ids.size(1), 1)
                )

        def collator(features):
            return {
                name: torch.tensor([feature[name] for feature in features])
                for name in ("input_ids", "attention_mask", "labels")
            }

        model = TinyModel()
        model.train()
        result = generate_pub_data_logits(
            {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
                "labels": [[-100, 2, 3]],
            },
            model,
            SimpleNamespace(
                metric_type="ce",
                top_k_strategy="highest",
                top_k_logits_keep=2,
            ),
            collator,
        )

        self.assertEqual(
            observations,
            {"training": False, "grad_enabled": False, "use_cache": False},
        )
        self.assertTrue(model.training)
        self.assertEqual(result[PER_STEP_INDICES][0, 0].tolist(), [5, 4])
        self.assertEqual(result[PER_STEP_LOGITS][0, 0].tolist(), [5.0, 4.0])
        self.assertEqual(result[PER_STEP_LOGITS].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
