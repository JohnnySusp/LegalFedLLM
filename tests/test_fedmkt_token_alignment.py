from __future__ import annotations

import inspect
import unittest


try:
    from shared.fedmkt_core.ml import token_alignment
except ModuleNotFoundError as exc:
    token_alignment = None
    ML_IMPORT_ERROR = exc
else:
    ML_IMPORT_ERROR = None


class BaseTokenizer:
    def __init__(self, tokens: dict[int, str]) -> None:
        self.tokens = tokens

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self.tokens[token_id] for token_id in token_ids]


class BlendingTokenizer(BaseTokenizer):
    pass


class UnregisteredBaseTokenizer(BaseTokenizer):
    pass


class UnregisteredBlendingTokenizer(BaseTokenizer):
    pass


@unittest.skipUnless(
    token_alignment is not None,
    f"optional ML dependencies are unavailable: {ML_IMPORT_ERROR}",
)
class FedMKTTokenAlignmentParityTests(unittest.TestCase):
    """Goldens generated from FATE-LLM commit 0c63377."""

    @classmethod
    def setUpClass(cls) -> None:
        token_alignment.TOKENIZER_TO_SPECIAL_TOKEN[BaseTokenizer] = "▁"
        token_alignment.TOKENIZER_TO_SPECIAL_TOKEN[BlendingTokenizer] = "Ġ"

    @classmethod
    def tearDownClass(cls) -> None:
        token_alignment.TOKENIZER_TO_SPECIAL_TOKEN.pop(BaseTokenizer, None)
        token_alignment.TOKENIZER_TO_SPECIAL_TOKEN.pop(
            BlendingTokenizer,
            None,
        )

    def test_unequal_length_dtw_golden(self) -> None:
        matches, cost, mapping_1, mapping_2, matrix = token_alignment.dtw(
            [0, 1],
            [0, 1, 2],
            norm_func=lambda left, right: abs(left - right),
        )

        self.assertEqual(matches, [(0, 0), (1, 1), (1, 2)])
        self.assertEqual(cost, 1.0)
        self.assertEqual(mapping_1, [[0], [1, 2]])
        self.assertEqual(mapping_2, [[0], [1], [1]])
        self.assertEqual(matrix.tolist(), [[0.0, 1.0, 3.0], [1.0, 0.0, 1.0]])

    def test_equal_cost_path_uses_upstream_tie_order(self) -> None:
        matches, cost, mapping_1, mapping_2, matrix = token_alignment.dtw(
            ["a", "b"],
            ["x", "y", "z"],
            norm_func=lambda _left, _right: 0,
        )

        self.assertEqual(matches, [(0, 0), (0, 1), (1, 2)])
        self.assertEqual(cost, 0.0)
        self.assertEqual(mapping_1, [[0, 1], [2]])
        self.assertEqual(mapping_2, [[0], [0], [1]])
        self.assertEqual(matrix.tolist(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def test_unicode_and_whitespace_marker_golden(self) -> None:
        distance = lambda left, right: token_alignment.token_levenshtein_distance(
            left,
            right,
            "▁",
            "Ġ",
        )
        matches, cost, mapping_1, mapping_2, matrix = token_alignment.dtw(
            ["▁Άρθρο", "ς"],
            ["ĠΆρθρο", "ς"],
            norm_func=distance,
        )

        self.assertEqual(matches, [(0, 0), (1, 1)])
        self.assertEqual(cost, 0.0)
        self.assertEqual(mapping_1, [[0], [1]])
        self.assertEqual(mapping_2, [[0], [1]])
        self.assertEqual(matrix.tolist(), [[0.0, 5.0], [5.0, 0.0]])

    def test_empty_sequences_are_rejected_explicitly(self) -> None:
        for series_1, series_2 in (([], [1]), ([1], []), ([], [])):
            with self.subTest(series_1=series_1, series_2=series_2):
                with self.assertRaisesRegex(
                    ValueError,
                    "two non-empty token sequences",
                ):
                    token_alignment.dtw(series_1, series_2)

    def test_one_to_one_logit_transform_golden(self) -> None:
        logits, indices = token_alignment.transform_step_logits(
            base_model_tokenizer=BaseTokenizer(
                {0: "▁Άρθρο", 1: "▁νόμος", 2: "▁δίκαιο"}
            ),
            blending_model_tokenizer=BlendingTokenizer(
                {10: "ĠΆρθρο", 11: "Ġνόμος", 12: "Ġδίκαιο"}
            ),
            base_model_vocab={"▁Άρθρο": 0, "▁νόμος": 1, "▁δίκαιο": 2},
            base_model_input_ids=[0, 1],
            blending_model_input_ids=[10, 11],
            blending_model_per_step_logits=[[0.7, 0.2], [0.6, 0.3]],
            blending_model_per_step_indices=[[10, 12], [11, 12]],
            blending_to_base_mapping={
                "▁Άρθρο": "▁Άρθρο",
                "▁νόμος": "▁νόμος",
                "▁δίκαιο": "▁δίκαιο",
            },
        )

        self.assertEqual(logits, [[0.7, 0.2], [0.6, 0.3]])
        self.assertEqual(indices, [[0, 2], [1, 2]])

    def test_one_to_many_logit_transform_uses_upstream_fallback(self) -> None:
        logits, indices = token_alignment.transform_step_logits(
            base_model_tokenizer=BaseTokenizer({0: "▁legal", 7: "▁legal"}),
            blending_model_tokenizer=BlendingTokenizer(
                {10: "Ġle", 11: "gal"}
            ),
            base_model_vocab={"▁legal": 7},
            base_model_input_ids=[0],
            blending_model_input_ids=[10, 11],
            blending_model_per_step_logits=[[0.8], [0.9]],
            blending_model_per_step_indices=[[10], [11]],
            blending_to_base_mapping={"▁le": "▁le", "gal": "gal"},
        )

        self.assertEqual(logits, [[1.0]])
        self.assertEqual(indices, [[7]])

    def test_profile_markers_do_not_require_a_class_registry_entry(self) -> None:
        logits, indices = token_alignment.transform_step_logits(
            base_model_tokenizer=UnregisteredBaseTokenizer({0: "Ġlaw"}),
            blending_model_tokenizer=UnregisteredBlendingTokenizer(
                {10: "Ġlaw"}
            ),
            base_model_vocab={"Ġlaw": 0},
            base_model_input_ids=[0],
            blending_model_input_ids=[10],
            blending_model_per_step_logits=[[0.9]],
            blending_model_per_step_indices=[[10]],
            blending_to_base_mapping={"Ġlaw": "Ġlaw"},
            base_model_special_token="Ġ",
            blending_model_special_token="Ġ",
        )

        self.assertEqual(logits, [[0.9]])
        self.assertEqual(indices, [[0]])

    def test_greedy_dp_implementation_is_removed(self) -> None:
        self.assertFalse(
            hasattr(token_alignment, "greedy_dynamic_matching")
        )
        parameter = inspect.signature(
            token_alignment.align_blending_model_logits_with_base_model_logits
        ).parameters["align_strategy"]
        self.assertEqual(parameter.default, "dtw")

        with self.assertRaisesRegex(ValueError, "not implemented"):
            token_alignment.transform_step_logits(
                base_model_tokenizer=BaseTokenizer({0: "▁law"}),
                blending_model_tokenizer=BlendingTokenizer({10: "Ġlaw"}),
                base_model_vocab={"▁law": 0},
                base_model_input_ids=[0],
                blending_model_input_ids=[10],
                blending_model_per_step_logits=[[0.9]],
                blending_model_per_step_indices=[[10]],
                blending_to_base_mapping={"▁law": "▁law"},
                align_strategy="greedy_dp",
            )


if __name__ == "__main__":
    unittest.main()
