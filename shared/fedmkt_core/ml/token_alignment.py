#
# NOTE: The dtw function is copied from FuseAI/FuseLLM
#       and the align_blending_model_logits_with_base_model_logits function is modified from FuseAI/FuseLLM
# Copyright FuseAI
#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import transformers
from rapidfuzz.distance import Levenshtein
import numpy as np

from typing import Dict, List
from shared.fedmkt_core.ml.special_token_mapping import TOKENIZER_TO_SPECIAL_TOKEN
from shared.fedmkt_core.ml.vars_define import (
    PER_STEP_LOGITS,
    PER_STEP_INDICES,
    ALIGNED_OTHER_LOGITS,
    ALIGNED_OTHER_INDICES,
    ALIGNED_OTHER_METRIC,
    METRIC
)


logger = logging.getLogger(__name__)


def token_levenshtein_distance(
    blending_token: str,
    base_token: str,
    blending_model_special_token: str,
    base_model_special_token: str,
) -> int:
    return Levenshtein.distance(
        blending_token.replace(blending_model_special_token, ""),
        base_token.replace(base_model_special_token, ""),
    )


def dtw(series_1, series_2, norm_func=np.linalg.norm):
    """code refer to: https://github.com/fanqiwan/FuseAI/blob/main/FuseLLM/src/utils/others.py#L318"""

    if len(series_1) == 0 or len(series_2) == 0:
        raise ValueError("DTW requires two non-empty token sequences")

    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(matrix[i, j + 1], matrix[i + 1, j], matrix[i, j])
    matrix = matrix[1:, 1:]
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    matches = []
    mappings_series_1 = [list() for v in range(matrix.shape[0])]
    mappings_series_2 = [list() for v in range(matrix.shape[1])]
    while i > 0 or j > 0:
        matches.append((i, j))
        mappings_series_1[i].append(j)
        mappings_series_2[j].append(i)
        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf
        move = np.argmin([option_diag, option_up, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    matches.append((0, 0))
    mappings_series_1[0].append(0)
    mappings_series_2[0].append(0)
    matches.reverse()
    for mp in mappings_series_1:
        mp.reverse()
    for mp in mappings_series_2:
        mp.reverse()

    return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix


def align_blending_model_logits_with_base_model_logits(base_examples,
                                                       indices,
                                                       blending_examples,
                                                       blending_to_base_mapping,
                                                       base_tokenizer,
                                                       blending_tokenizer,
                                                       blending_model_index,
                                                       skip_align=False,
                                                       align_strategy="dtw"):
    """modified from https://github.com/fanqiwan/FuseAI/blob/main/FuseLLM/src/utils/token_alignment.py#L101"""
    base_features = [{key: base_examples[key][i] for key in base_examples} for i in
                     range(len(base_examples[next(iter(base_examples))]))]
    blending_features = [blending_examples[idx] for idx in indices]
    aligned_per_step_logits_list, aligned_per_step_indices_list = [], []
    per_step_logits_list, per_step_indices_list = [], []
    metric_ce_aligned = []
    for base_feature, blending_feature in zip(base_features, blending_features):
        base_feature[PER_STEP_LOGITS] = base_feature[PER_STEP_LOGITS][:len(base_feature['input_ids'])]
        base_feature[PER_STEP_INDICES] = base_feature[PER_STEP_INDICES][:len(base_feature['input_ids'])]
        blending_feature[PER_STEP_LOGITS] = blending_feature[PER_STEP_LOGITS][:len(blending_feature['input_ids'])]
        blending_feature[PER_STEP_INDICES] = blending_feature[PER_STEP_INDICES][:len(blending_feature['input_ids'])]
        if skip_align is True:
            aligned_blending_model_per_step_logits = blending_feature[PER_STEP_LOGITS]
            aligned_blending_model_per_step_indices = blending_feature[PER_STEP_INDICES]
        else:
            aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices = transform_step_logits(
                base_model_tokenizer=base_tokenizer,
                blending_model_tokenizer=blending_tokenizer,
                base_model_vocab=base_tokenizer.get_vocab(),
                base_model_input_ids=base_feature['input_ids'],
                blending_model_input_ids=blending_feature['input_ids'],
                blending_model_per_step_logits=blending_feature[PER_STEP_LOGITS],
                blending_model_per_step_indices=blending_feature[PER_STEP_INDICES],
                blending_to_base_mapping=blending_to_base_mapping,
                align_strategy=align_strategy
            )

        aligned_per_step_logits_list.append(aligned_blending_model_per_step_logits)
        aligned_per_step_indices_list.append(aligned_blending_model_per_step_indices)
        per_step_logits_list.append(base_feature[PER_STEP_LOGITS])
        per_step_indices_list.append(base_feature[PER_STEP_INDICES])
        metric_ce_aligned.append(blending_feature[METRIC])

    base_examples[PER_STEP_LOGITS] = per_step_logits_list
    base_examples[PER_STEP_INDICES] = per_step_indices_list
    base_examples[f"{ALIGNED_OTHER_LOGITS}_{blending_model_index}"] = aligned_per_step_logits_list
    base_examples[f"{ALIGNED_OTHER_INDICES}_{blending_model_index}"] = aligned_per_step_indices_list
    base_examples[f"{ALIGNED_OTHER_METRIC}_{blending_model_index}"] = metric_ce_aligned

    return base_examples


def transform_step_logits(base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
                          blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
                          base_model_vocab: Dict[str, int],
                          base_model_input_ids: List[int],
                          blending_model_input_ids: List[int],
                          blending_model_per_step_logits: List[List[float]],
                          blending_model_per_step_indices: List[List[int]],
                          blending_to_base_mapping: Dict[str, str] = None,
                          align_strategy: str = "dtw"
                          ):
    """modified from https://github.com/fanqiwan/FuseAI/blob/main/FuseLLM/src/utils/others.py#L364"""
    """Align blending model per step logits & indices with base model."""
    base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
    blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(blending_model_input_ids)
    base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[base_model_tokenizer.__class__]
    blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[blending_model_tokenizer.__class__]

    aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices = [], []
    if align_strategy == "dtw":
        def dist_fn(a, b):
            """Calculate Levenshtein distance between blending and base tokens."""
            return token_levenshtein_distance(
                a,
                b,
                blending_model_special_token,
                base_model_special_token,
            )

        _, _, _, base_to_blending, _ = dtw(blending_model_tokens, base_model_tokens, norm_func=dist_fn)
        for i, blending_idx in enumerate(base_to_blending):
            aligned_blending_model_per_step_logit = []
            aligned_blending_model_per_step_index = []
            if len(blending_idx) == 1:  # one base token map to one blending token
                j = blending_idx[0]
                base_token = base_model_tokens[i]
                blending_token = blending_model_tokens[j].replace(blending_model_special_token,
                                                                  base_model_special_token)
                if (
                    blending_model_tokenizer.__class__ == transformers.GPTNeoXTokenizerFast
                    or blending_model_tokenizer.__class__ == transformers.GPT2TokenizerFast) and i == 0 and base_token.startswith(
                    base_model_special_token) and not blending_token.startswith(base_model_special_token):
                    blending_token = base_model_special_token + blending_token  # special case for mpt

                if (base_token == blending_token) or (
                        blending_token in blending_to_base_mapping and base_token == blending_to_base_mapping[
                    blending_token]):  # find the aligned mapping, use the corresponding logits
                    # the logits and indices at this step
                    for blending_logit, blending_index in zip(blending_model_per_step_logits[j],
                                                              blending_model_per_step_indices[j]):
                        # the token corresponds to the logit and indices
                        blending_t = blending_model_tokenizer.convert_ids_to_tokens([blending_index])[0].replace(
                            blending_model_special_token, base_model_special_token)
                        blending_t = blending_to_base_mapping[blending_t]
                        if blending_t in base_model_vocab:
                            aligned_index = base_model_vocab[blending_t]  # the index of the token in base model vocab
                            if aligned_index not in aligned_blending_model_per_step_index:
                                aligned_blending_model_per_step_index.append(aligned_index)
                                aligned_blending_model_per_step_logit.append(blending_logit)
                        else:
                            logger.warning(f"blending_t: {blending_t} not in base_model_vocab!")
                else:  # find error aligned mapping, use the one-hot logits
                    aligned_blending_model_per_step_index.append(base_model_vocab[base_token])
                    aligned_blending_model_per_step_logit.append(1.0)
            else:  # one base token map to multiple blending token, in this case only fit base token. use the one-hot logits
                base_token = base_model_tokens[i]
                aligned_blending_model_per_step_index.append(base_model_vocab[base_token])
                aligned_blending_model_per_step_logit.append(1.0)
            aligned_blending_model_per_step_indices.append(aligned_blending_model_per_step_index)
            aligned_blending_model_per_step_logits.append(aligned_blending_model_per_step_logit)
    else:
        raise ValueError(f"{align_strategy} not implemented yet.")

    return aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices


def token_align(
    base_model_logits_datasets,
    blending_model_logits_dataset,
    base_tokenizer,
    blending_tokenizer,
    blending_to_base_mapping,
    blending_model_index,
    batch_size=4,
    preprocessing_num_workers=4,
    skip_align=False,
    align_strategy="dtw",
):
    assert len(base_model_logits_datasets) == len(blending_model_logits_dataset)
    base_model_blending_model_logits_datasets = base_model_logits_datasets.map(
        align_blending_model_logits_with_base_model_logits,
        batched=True,
        batch_size=batch_size,
        with_indices=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=True,
        fn_kwargs={"blending_examples": blending_model_logits_dataset,
                   "blending_to_base_mapping": blending_to_base_mapping,
                   "base_tokenizer": base_tokenizer,
                   "blending_tokenizer": blending_tokenizer,
                   "blending_model_index": blending_model_index,
                   "skip_align": skip_align,
                   "align_strategy": align_strategy},
        keep_in_memory=True,
        desc="Align blending model's logits with base model's logits.",
    )

    return base_model_blending_model_logits_datasets
