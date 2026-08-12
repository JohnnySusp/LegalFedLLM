from __future__ import annotations

import importlib.util
import types
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def fixed_model():
    import torch

    class FixedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_parameter("anchor", torch.nn.Parameter(torch.zeros(())))

        def forward(self, input_ids, attention_mask, labels):
            logits = torch.tensor(
                [[[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [4.0, 0.0]]],
                device=input_ids.device,
            )
            return {"loss": self.anchor * 0, "logits": logits}

    return FixedModel()


def trainer():
    from shared.fedmkt_core.ml.trainer import FedMKTTrainer

    value = FedMKTTrainer.__new__(FedMKTTrainer)
    value.label_smoother = None
    value.args = types.SimpleNamespace(past_index=-1)
    value.blending_num = 0
    value.distill_loss_type = "ce"
    value.lm_loss_weight = 0.0
    value.distill_strategy = "greater"
    return value


def inputs(targets: list[int], labels: list[int] | None = None) -> dict:
    import torch

    from shared.fedmkt_core.ml.vars_define import METRIC, SELF_TARGET_DIST

    target_dist = torch.zeros(1, 4, 2)
    for position, token_id in enumerate(targets):
        target_dist[0, position, token_id] = 1.0
    values = {
        "input_ids": torch.tensor([[0, 0, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        SELF_TARGET_DIST: target_dist,
        METRIC: [0.1],
    }
    if labels is not None:
        values["labels"] = torch.tensor([labels])
    return values


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for FedMKT trainer tests")
class FedMKTTrainerMaskTests(unittest.TestCase):
    def test_prompt_and_final_positions_do_not_change_distillation_loss(self) -> None:
        import torch

        model = fixed_model()
        answer_labels = [-100, -100, 0, 0]

        first = trainer().compute_loss(
            model,
            inputs([0, 0, 0, 0], answer_labels),
        )
        changed_excluded_positions = trainer().compute_loss(
            model,
            inputs([1, 0, 0, 1], answer_labels),
        )
        changed_answer_position = trainer().compute_loss(
            model,
            inputs([0, 1, 0, 0], answer_labels),
        )

        torch.testing.assert_close(first, changed_excluded_positions)
        self.assertGreater(changed_answer_position.item(), first.item())

    def test_labels_and_supervised_answer_positions_are_required(self) -> None:
        model = fixed_model()
        with self.assertRaisesRegex(ValueError, "requires labels"):
            trainer().compute_loss(model, inputs([0, 0, 0, 0]))
        with self.assertRaisesRegex(ValueError, "supervised target token"):
            trainer().compute_loss(
                model,
                inputs([0, 0, 0, 0], [-100, -100, -100, -100]),
            )


if __name__ == "__main__":
    unittest.main()
