from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
            full_logsumexp=[float(index) + 1.0 for index in range(9)],
            gold_token_ids=[0, *([-100] * 8)],
            gold_token_logits=[0.75, *([0.0] * 8)],
            gold_token_nll=[0.25, *([0.0] * 8)],
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
        from shared.fedmkt_core.ml.vars_define import (
            FULL_LOGSUMEXP,
            GOLD_TOKEN_IDS,
            GOLD_TOKEN_LOGITS,
            GOLD_TOKEN_NLL,
            METRIC,
            PER_STEP_INDICES,
            PER_STEP_LOGITS,
        )

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
        expected_lse = torch.logsumexp(torch.arange(6, dtype=torch.float32), dim=0)
        torch.testing.assert_close(
            result[FULL_LOGSUMEXP][0],
            expected_lse.repeat(3),
        )
        self.assertEqual(result[GOLD_TOKEN_IDS][0].tolist(), [2, 3, -100])
        torch.testing.assert_close(
            result[GOLD_TOKEN_LOGITS][0],
            torch.tensor([2.0, 3.0, 0.0]),
        )
        torch.testing.assert_close(
            result[GOLD_TOKEN_NLL][0],
            torch.tensor(
                [float(expected_lse - 2.0), float(expected_lse - 3.0), 0.0]
            ),
        )
        torch.testing.assert_close(
            result[METRIC],
            torch.tensor([float(expected_lse - 2.5)]),
        )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for FedMKT metric tests")
class FedMKTSequenceChunkTests(unittest.TestCase):
    @staticmethod
    def _collator(features):
        import torch

        return {
            name: torch.tensor([feature[name] for feature in features])
            for name in ("input_ids", "attention_mask", "labels")
        }

    @staticmethod
    def _inputs():
        return {
            "input_ids": [[1, 2, 3, 4, 5, 6]],
            "attention_mask": [[1, 1, 1, 1, 1, 1]],
            "labels": [[-100, -100, 3, 4, 5, 6]],
        }

    @staticmethod
    def _arguments():
        return SimpleNamespace(
            metric_type="ce",
            top_k_strategy="highest",
            top_k_logits_keep=3,
        )

    def test_sequence_chunked_generation_matches_existing_evidence(self) -> None:
        import torch

        from shared.fedmkt_core.ml.logit_generation import generate_pub_data_logits
        from shared.fedmkt_core.ml.vars_define import (
            FULL_LOGSUMEXP,
            GOLD_TOKEN_IDS,
            GOLD_TOKEN_LOGITS,
            GOLD_TOKEN_NLL,
            METRIC,
            PER_STEP_INDICES,
            PER_STEP_LOGITS,
        )

        class TinyCachedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.cached_calls: list[tuple[list[int], int]] = []

            def forward(
                self,
                input_ids,
                attention_mask,
                use_cache,
                past_key_values=None,
                cache_position=None,
            ):
                if past_key_values is None:
                    past = input_ids[:, :0]
                else:
                    past = past_key_values
                if use_cache:
                    expected = torch.arange(
                        past.size(1),
                        past.size(1) + input_ids.size(1),
                        device=input_ids.device,
                    )
                    torch.testing.assert_close(cache_position, expected)
                    self.cached_calls.append(
                        (cache_position.tolist(), attention_mask.size(1))
                    )
                full = torch.cat((past, input_ids), dim=1)
                vocabulary = torch.arange(
                    9,
                    dtype=torch.float32,
                    device=input_ids.device,
                ).view(1, 1, -1)
                rows = []
                for local_index in range(input_ids.size(1)):
                    global_index = past.size(1) + local_index
                    prefix_sum = full[:, : global_index + 1].sum(
                        dim=1,
                        keepdim=True,
                    ).float()
                    position = torch.tensor(
                        float(global_index),
                        device=input_ids.device,
                    )
                    rows.append(
                        prefix_sum.unsqueeze(-1) * 0.03
                        + position * 0.17
                        + vocabulary * 0.29
                        + ((position + vocabulary) % 4) * 0.07
                    )
                logits = torch.cat(rows, dim=1)
                return SimpleNamespace(
                    logits=logits,
                    past_key_values=full.detach() if use_cache else None,
                )

        full_model = TinyCachedModel()
        full = generate_pub_data_logits(
            self._inputs(),
            full_model,
            self._arguments(),
            self._collator,
        )
        self.assertEqual(full_model.cached_calls, [])

        chunked_model = TinyCachedModel()
        chunked = generate_pub_data_logits(
            self._inputs(),
            chunked_model,
            self._arguments(),
            self._collator,
            sequence_chunk_size=2,
        )
        self.assertEqual(
            chunked_model.cached_calls,
            [([0, 1], 2), ([2, 3], 4), ([4, 5], 6)],
        )
        self.assertTrue(
            torch.equal(full[PER_STEP_INDICES], chunked[PER_STEP_INDICES])
        )
        self.assertTrue(
            torch.equal(full[GOLD_TOKEN_IDS], chunked[GOLD_TOKEN_IDS])
        )
        for key in (
            PER_STEP_LOGITS,
            FULL_LOGSUMEXP,
            GOLD_TOKEN_LOGITS,
            GOLD_TOKEN_NLL,
            METRIC,
        ):
            torch.testing.assert_close(
                chunked[key],
                full[key],
                rtol=0,
                atol=1e-6,
            )

    def test_negative_sequence_chunk_size_is_rejected(self) -> None:
        import torch

        from shared.fedmkt_core.ml.logit_generation import generate_pub_data_logits

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, input_ids, attention_mask, use_cache):
                return SimpleNamespace(
                    logits=torch.zeros(input_ids.size(0), input_ids.size(1), 9)
                )

        with self.assertRaisesRegex(ValueError, "sequence_chunk_size"):
            generate_pub_data_logits(
                self._inputs(),
                TinyModel(),
                self._arguments(),
                self._collator,
                sequence_chunk_size=-1,
            )

    def test_sequence_chunking_rejects_multi_sample_batch(self) -> None:
        import torch

        from shared.fedmkt_core.ml.logit_generation import generate_pub_data_logits

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, input_ids, attention_mask, use_cache):
                return SimpleNamespace(
                    logits=torch.zeros(input_ids.size(0), input_ids.size(1), 9)
                )

        inputs = self._inputs()
        inputs = {key: value * 2 for key, value in inputs.items()}
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            generate_pub_data_logits(
                inputs,
                TinyModel(),
                self._arguments(),
                self._collator,
                sequence_chunk_size=2,
            )


class KnowledgeSequenceChunkConfigurationTests(unittest.TestCase):
    def test_backend_defaults_off_and_reads_explicit_chunk_size(self) -> None:
        from client.peft_backend import TransformersPeftTrainingBackend

        profile = pinned_client_profile(QWEN_PROFILE_ID)
        execution = SimpleNamespace(backend="transformers")
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "CLIENT_KNOWLEDGE_BATCH_SIZE": "1",
                    "CLIENT_KNOWLEDGE_SEQUENCE_CHUNK_SIZE": "0",
                },
                clear=False,
            ):
                backend = TransformersPeftTrainingBackend(
                    data_dir=directory,
                    model_profile=profile,
                    execution_profile=execution,
                )
                self.assertEqual(backend.knowledge_sequence_chunk_size, 0)

            with patch.dict(
                os.environ,
                {
                    "CLIENT_KNOWLEDGE_BATCH_SIZE": "1",
                    "CLIENT_KNOWLEDGE_SEQUENCE_CHUNK_SIZE": "64",
                },
                clear=False,
            ):
                backend = TransformersPeftTrainingBackend(
                    data_dir=directory,
                    model_profile=profile,
                    execution_profile=execution,
                )
                self.assertEqual(backend.knowledge_sequence_chunk_size, 64)

    def test_backend_requires_batch_one_when_sequence_chunking_is_enabled(self) -> None:
        from client.peft_backend import TransformersPeftTrainingBackend

        profile = pinned_client_profile(QWEN_PROFILE_ID)
        execution = SimpleNamespace(backend="transformers")
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "CLIENT_KNOWLEDGE_BATCH_SIZE": "2",
                    "CLIENT_KNOWLEDGE_SEQUENCE_CHUNK_SIZE": "64",
                },
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "BATCH_SIZE=1"):
                    TransformersPeftTrainingBackend(
                        data_dir=directory,
                        model_profile=profile,
                        execution_profile=execution,
                    )


if __name__ == "__main__":
    unittest.main()
