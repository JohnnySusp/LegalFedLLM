from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for sparse-target tests")
class SparseTargetTests(unittest.TestCase):
    def _build(self, **overrides):
        import torch

        from shared.fedmkt_core.ml.sparse_targets import build_sparse_target_batch

        values = {
            "token_id_rows": [
                [[4, 1, 3], [2, 5, 0]],
                [[1, 6, 2]],
            ],
            "logit_rows": [
                [[3.0, 1.0, -1.0], [2.0, 0.5, -0.5]],
                [[1.5, 0.25, -2.0]],
            ],
            "max_length": 2,
            "top_k": 3,
            "vocab_size": 7,
            "pad_token_id": 0,
            "temperature": 1.5,
            "dtype": torch.float32,
        }
        values.update(overrides)
        return build_sparse_target_batch(**values)

    def test_sparse_ce_and_kl_match_dense_oracle_and_gradients(self) -> None:
        import torch
        from torch.nn.functional import cross_entropy, kl_div, log_softmax

        from shared.fedmkt_core.ml.sparse_targets import (
            answer_only_sparse_distillation_loss,
            densify_sparse_targets_for_test,
        )

        targets = self._build()
        dense_targets = densify_sparse_targets_for_test(
            targets,
            vocab_size=7,
        )
        expected_dense_targets = torch.zeros(2, 2, 7)
        expected_dense_targets[0, 0, [4, 1, 3]] = torch.softmax(
            torch.tensor([3.0, 1.0, -1.0]) / 1.5,
            dim=-1,
        )
        expected_dense_targets[0, 1, [2, 5, 0]] = torch.softmax(
            torch.tensor([2.0, 0.5, -0.5]) / 1.5,
            dim=-1,
        )
        expected_dense_targets[1, 0, [1, 6, 2]] = torch.softmax(
            torch.tensor([1.5, 0.25, -2.0]) / 1.5,
            dim=-1,
        )
        expected_dense_targets[1, 1, 0] = 1.0
        torch.testing.assert_close(dense_targets, expected_dense_targets)

        labels = torch.tensor([[-100, 2], [-100, 1]])
        attention_mask = torch.tensor([[1, 1], [1, 0]])
        base_logits = torch.tensor(
            [
                [[0.2, 0.1, -0.3, 0.4, 0.8, -0.2, 0.0],
                 [0.0, 0.3, 0.7, -0.1, 0.5, 0.2, -0.4]],
                [[0.1, 0.4, -0.2, 0.8, 0.0, -0.5, 0.3],
                 [0.6, 0.1, -0.4, 0.2, 0.0, -0.3, 0.5]],
            ],
            dtype=torch.float32,
        )
        distillation_mask = labels[..., 1:].ne(-100) & attention_mask[..., 1:].bool()

        for loss_type in ("ce", "kl"):
            sparse_logits = base_logits.clone().requires_grad_(True)
            sparse_loss = answer_only_sparse_distillation_loss(
                sparse_logits,
                targets,
                labels=labels,
                attention_mask=attention_mask,
                loss_type=loss_type,
            )
            sparse_loss.backward()

            dense_logits = base_logits.clone().requires_grad_(True)
            if loss_type == "ce":
                per_position = cross_entropy(
                    dense_logits.view(-1, 7),
                    dense_targets.view(-1, 7),
                    reduction="none",
                ).view(2, 2)
            else:
                per_position = kl_div(
                    log_softmax(dense_logits, dim=-1),
                    dense_targets,
                    log_target=False,
                    reduction="none",
                ).sum(dim=-1)
            dense_loss = (
                per_position[..., :-1] * distillation_mask
            ).sum() / distillation_mask.sum()
            dense_loss.backward()

            torch.testing.assert_close(sparse_loss, dense_loss)
            torch.testing.assert_close(sparse_logits.grad, dense_logits.grad)

    @unittest.skipUnless(
        TRANSFORMERS_AVAILABLE,
        "Transformers is required for inherited dense-collator parity",
    )
    def test_sparse_targets_match_inherited_dense_collator(self) -> None:
        import torch

        from shared.fedmkt_core.ml.data_collator import DataCollatorForFedMKT
        from shared.fedmkt_core.ml.sparse_targets import (
            densify_sparse_targets_for_test,
        )
        from shared.fedmkt_core.ml.vars_define import (
            PER_STEP_INDICES,
            PER_STEP_LOGITS,
            SELF_TARGET_DIST,
        )

        class DummyTokenizer:
            pad_token_id = 0
            padding_side = "right"

            def get_vocab(self):
                return {str(token_id): token_id for token_id in range(7)}

            def pad(self, features, *args, **kwargs):
                del args
                max_length = kwargs["max_length"]
                values = {}
                for name in ("input_ids", "attention_mask"):
                    rows = []
                    for feature in features:
                        row = list(feature[name])
                        fill = 0
                        rows.append(row + [fill] * (max_length - len(row)))
                    values[name] = torch.tensor(rows)
                return values

        features = [
            {
                "input_ids": [4, 2],
                "attention_mask": [1, 1],
                "labels": [-100, 2],
                PER_STEP_INDICES: [[4, 1, 3], [2, 5, 0]],
                PER_STEP_LOGITS: [[3.0, 1.0, -1.0], [2.0, 0.5, -0.5]],
            },
            {
                "input_ids": [1],
                "attention_mask": [1],
                "labels": [-100],
                PER_STEP_INDICES: [[1, 6, 2]],
                PER_STEP_LOGITS: [[1.5, 0.25, -2.0]],
            },
        ]
        dense_batch = DataCollatorForFedMKT(
            tokenizer=DummyTokenizer(),
            padding="max_length",
            max_length=2,
            blending_num=0,
            vocab_size=7,
            dtype=torch.float32,
            distill_temperature=1.5,
        )(features)
        sparse_dense = densify_sparse_targets_for_test(
            self._build(),
            vocab_size=7,
        )

        torch.testing.assert_close(dense_batch[SELF_TARGET_DIST], sparse_dense)

    def test_duplicate_mapped_ids_keep_first_before_softmax(self) -> None:
        import torch

        targets = self._build(
            token_id_rows=[[[4, 4, 2]]],
            logit_rows=[[[3.0, 1.0, 0.0]]],
            max_length=1,
        )

        self.assertEqual(targets.token_ids.tolist(), [[[4, 2, 0]]])
        self.assertEqual(targets.valid_mask.tolist(), [[[True, True, False]]])
        torch.testing.assert_close(
            targets.probabilities[0, 0, :2],
            torch.softmax(torch.tensor([3.0, 0.0]) / 1.5, dim=-1),
        )
        self.assertEqual(targets.probabilities[0, 0, 2].item(), 0.0)

    def test_empty_aligned_row_uses_base_fallback(self) -> None:
        import torch

        targets = self._build(
            token_id_rows=[[[]]],
            logit_rows=[[[]]],
            fallback_token_id_rows=[[[5, 1]]],
            fallback_logit_rows=[[[2.0, 0.0]]],
            max_length=1,
        )

        self.assertEqual(targets.token_ids.tolist(), [[[5, 1, 0]]])
        torch.testing.assert_close(
            targets.probabilities[0, 0, :2],
            torch.softmax(torch.tensor([2.0, 0.0]) / 1.5, dim=-1),
        )

    def test_one_hot_and_padding_targets_match_upstream_behavior(self) -> None:
        targets = self._build(
            token_id_rows=[[[6]]],
            logit_rows=[[[1.0]]],
            max_length=3,
        )

        self.assertEqual(
            targets.valid_mask.tolist(),
            [[[True, False, False], [True, False, False], [True, False, False]]],
        )
        self.assertEqual(targets.token_ids[:, :, 0].tolist(), [[6, 0, 0]])
        self.assertEqual(targets.probabilities[:, :, 0].tolist(), [[1.0, 1.0, 1.0]])

    def test_answer_only_causal_mask_excludes_prompt_and_final_positions(self) -> None:
        import torch

        from shared.fedmkt_core.ml.sparse_targets import (
            answer_only_sparse_distillation_loss,
        )

        model_logits = torch.tensor(
            [[[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [4.0, 0.0]]]
        )
        labels = torch.tensor([[-100, -100, 0, 0]])
        attention_mask = torch.ones_like(labels)

        first = self._build(
            token_id_rows=[[[0], [0], [0], [0]]],
            logit_rows=[[[1.0], [1.0], [1.0], [1.0]]],
            max_length=4,
            top_k=1,
            vocab_size=2,
        )
        excluded_changed = self._build(
            token_id_rows=[[[1], [0], [0], [1]]],
            logit_rows=[[[1.0], [1.0], [1.0], [1.0]]],
            max_length=4,
            top_k=1,
            vocab_size=2,
        )
        answer_changed = self._build(
            token_id_rows=[[[0], [1], [0], [0]]],
            logit_rows=[[[1.0], [1.0], [1.0], [1.0]]],
            max_length=4,
            top_k=1,
            vocab_size=2,
        )

        original_loss = answer_only_sparse_distillation_loss(
            model_logits,
            first,
            labels=labels,
            attention_mask=attention_mask,
        )
        excluded_loss = answer_only_sparse_distillation_loss(
            model_logits,
            excluded_changed,
            labels=labels,
            attention_mask=attention_mask,
        )
        changed_loss = answer_only_sparse_distillation_loss(
            model_logits,
            answer_changed,
            labels=labels,
            attention_mask=attention_mask,
        )

        torch.testing.assert_close(original_loss, excluded_loss)
        self.assertGreater(changed_loss.item(), original_loss.item())

    def test_invalid_rows_and_tampered_targets_fail_explicitly(self) -> None:
        import torch

        from shared.fedmkt_core.ml.sparse_targets import (
            SparseTargetBatch,
            SparseTargetError,
            validate_sparse_target_batch,
        )

        invalid_builds = (
            {"token_id_rows": [[[7]]], "logit_rows": [[[1.0]]]},
            {"token_id_rows": [[[1]]], "logit_rows": [[[float("nan")]]]},
            {"token_id_rows": [[[1, 2]]], "logit_rows": [[[1.0]]]},
            {"token_id_rows": [[[]]], "logit_rows": [[[]]], "max_length": 1},
            {"temperature": 0.0},
        )
        for overrides in invalid_builds:
            with self.subTest(overrides=overrides):
                with self.assertRaises(SparseTargetError):
                    self._build(**overrides)

        valid = self._build()
        tampered_probabilities = valid.probabilities.clone()
        tampered_probabilities[0, 0, 0] = 0.0
        with self.assertRaisesRegex(SparseTargetError, "sum to one"):
            validate_sparse_target_batch(
                SparseTargetBatch(
                    token_ids=valid.token_ids,
                    probabilities=tampered_probabilities,
                    valid_mask=valid.valid_mask,
                ),
                vocab_size=7,
            )

        duplicate_ids = valid.token_ids.clone()
        duplicate_ids[0, 0, 1] = duplicate_ids[0, 0, 0]
        with self.assertRaisesRegex(SparseTargetError, "unique"):
            validate_sparse_target_batch(
                SparseTargetBatch(
                    token_ids=duplicate_ids,
                    probabilities=valid.probabilities,
                    valid_mask=valid.valid_mask,
                ),
                vocab_size=7,
            )

    def test_storage_scales_with_top_k_not_vocabulary(self) -> None:
        from shared.fedmkt_core.ml.sparse_targets import (
            SparseTargetError,
            densify_sparse_targets_for_test,
        )

        qwen_targets = self._build(vocab_size=151936)
        granite_targets = self._build(vocab_size=49159)

        self.assertEqual(qwen_targets.token_ids.shape, (2, 2, 3))
        self.assertEqual(granite_targets.token_ids.shape, (2, 2, 3))
        self.assertEqual(qwen_targets.token_ids.numel(), 12)
        self.assertEqual(granite_targets.token_ids.numel(), 12)
        with self.assertRaisesRegex(SparseTargetError, "fixture-only"):
            densify_sparse_targets_for_test(
                qwen_targets,
                vocab_size=151936,
                maximum_elements=100,
            )


if __name__ == "__main__":
    unittest.main()
