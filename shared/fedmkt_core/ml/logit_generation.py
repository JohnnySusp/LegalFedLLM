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
# Adapted for LegalFedLLM from FATE-LLM commit
# 0c63377e468f0f62a9bdf5fb32424688b9478553.

from __future__ import annotations

import gc

import torch
import torch.nn.functional as F

from shared.fedmkt_core.ml.vars_define import (
    FULL_LOGSUMEXP,
    GOLD_TOKEN_IDS,
    GOLD_TOKEN_LOGITS,
    GOLD_TOKEN_NLL,
    METRIC,
    PER_STEP_INDICES,
    PER_STEP_LOGITS,
)


class Metric:
    @classmethod
    def cal_metric(cls, logits, input_ids, attention_mask, labels, training_args):
        if training_args.metric_type == "ce":
            return cls.cal_ce(logits, input_ids, attention_mask, labels, training_args)
        raise NotImplementedError(
            f"metric={training_args.metric_type} is not implemented"
        )

    @classmethod
    def cal_ce(cls, logits, input_ids, attention_mask, labels, training_args):
        shifted_labels = labels[..., 1:].contiguous()
        metric = F.cross_entropy(
            logits[..., :-1, :].contiguous().float().view(-1, logits.size(-1)),
            shifted_labels.view(-1),
            reduction="none",
        ).view(logits.size(0), -1)
        mask = shifted_labels.ne(-100) & attention_mask[..., 1:].bool()
        supervised = mask.sum(dim=-1)
        if torch.any(supervised == 0):
            raise ValueError("CE requires at least one supervised target token")
        result = (metric * mask).sum(dim=-1) / supervised
        if not torch.isfinite(result).all():
            raise ValueError("CE produced a non-finite per-sample loss")
        return result


class LogitsSelection:
    @classmethod
    def select_logits(cls, logits, training_args):
        if training_args.top_k_strategy == "highest":
            return cls.select_highest(logits, training_args.top_k_logits_keep)
        raise NotImplementedError(
            f"logits selection strategy={training_args.top_k_strategy} is not implemented"
        )

    @classmethod
    def select_highest(cls, logits, top_k_logits_keep):
        return torch.topk(logits, k=top_k_logits_keep, dim=-1)


def generate_pub_data_logits(
    inputs,
    model,
    training_args,
    data_collator,
    *,
    sequence_chunk_size=0,
):
    input_keys = ["attention_mask", "input_ids", "labels"]
    inputs_per_batched = [dict() for _ in range(len(inputs["input_ids"]))]
    for key in input_keys:
        if key not in inputs:
            continue
        for index, value in enumerate(inputs[key]):
            inputs_per_batched[index][key] = value

    if "attention_mask" not in inputs:
        for item in inputs_per_batched:
            item["attention_mask"] = [1] * len(item["input_ids"])

    batch = data_collator(inputs_per_batched)
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    if training_args.top_k_logits_keep is None:
        raise ValueError("top_k_logits_keep must be configured")
    sequence_chunk_size = int(sequence_chunk_size)
    if sequence_chunk_size < 0:
        raise ValueError("sequence_chunk_size must be non-negative")
    if sequence_chunk_size > 0 and input_ids.size(0) != 1:
        raise ValueError(
            "sequence-chunked knowledge generation requires batch size 1"
        )

    was_training = model.training
    model.eval()
    with torch.no_grad():
        if sequence_chunk_size == 0:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            metric = Metric.cal_metric(
                logits, input_ids, attention_mask, labels, training_args
            )
            selected_logits, selected_indices = LogitsSelection.select_logits(
                logits, training_args
            )
            float_logits = logits.float()
            full_logsumexp = torch.logsumexp(float_logits, dim=-1)
            gold_token_ids = torch.full_like(input_ids, -100)
            gold_token_ids[..., :-1] = labels[..., 1:]
            gold_token_logits = torch.zeros_like(full_logsumexp)
            gold_token_nll = torch.zeros_like(full_logsumexp)
            supervised = gold_token_ids.ne(-100)
            safe_gold_ids = gold_token_ids.clamp_min(0)
            gold_logits = float_logits.gather(
                dim=-1,
                index=safe_gold_ids.unsqueeze(-1),
            ).squeeze(-1)
            gold_token_logits[supervised] = gold_logits[supervised]
            gold_token_nll[supervised] = (
                full_logsumexp[supervised] - gold_logits[supervised]
            )
            inputs[PER_STEP_LOGITS] = selected_logits.detach().float().cpu()
            inputs[PER_STEP_INDICES] = selected_indices.detach().cpu()
            inputs[FULL_LOGSUMEXP] = full_logsumexp.detach().cpu()
            inputs[GOLD_TOKEN_IDS] = gold_token_ids.detach().cpu()
            inputs[GOLD_TOKEN_LOGITS] = gold_token_logits.detach().cpu()
            inputs[GOLD_TOKEN_NLL] = gold_token_nll.detach().cpu()
            inputs[METRIC] = metric.detach().cpu()
            del (
                float_logits,
                full_logsumexp,
                gold_logits,
                gold_token_ids,
                gold_token_logits,
                gold_token_nll,
                logits,
                metric,
                safe_gold_ids,
                selected_indices,
                selected_logits,
                supervised,
            )
        else:
            sequence_length = input_ids.size(1)
            gold_token_ids = torch.full_like(input_ids, -100)
            gold_token_ids[..., :-1] = labels[..., 1:]
            metric_mask = torch.zeros_like(gold_token_ids, dtype=torch.bool)
            metric_mask[..., :-1] = (
                gold_token_ids[..., :-1].ne(-100)
                & attention_mask[..., 1:].bool()
            )
            supervised_count = metric_mask.sum(dim=-1)
            if torch.any(supervised_count == 0):
                raise ValueError("CE requires at least one supervised target token")

            metric_sum = torch.zeros(
                input_ids.size(0),
                dtype=torch.float32,
                device=device,
            )
            selected_logit_chunks = []
            selected_index_chunks = []
            full_logsumexp_chunks = []
            gold_token_logit_chunks = []
            gold_token_nll_chunks = []
            past_key_values = None

            for start in range(0, sequence_length, sequence_chunk_size):
                stop = min(start + sequence_chunk_size, sequence_length)
                cache_position = torch.arange(start, stop, device=device)
                output = model(
                    input_ids=input_ids[..., start:stop],
                    attention_mask=attention_mask[..., :stop],
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                )
                logits = output.logits
                past_key_values = output.past_key_values
                if past_key_values is None:
                    raise RuntimeError(
                        "sequence-chunked knowledge generation requires a KV cache"
                    )
                if logits.size(1) != stop - start:
                    raise RuntimeError(
                        "model returned an unexpected cached sequence length"
                    )

                selected_logits, selected_indices = LogitsSelection.select_logits(
                    logits, training_args
                )
                float_logits = logits.float()
                full_logsumexp = torch.logsumexp(float_logits, dim=-1)

                gold_ids = gold_token_ids[..., start:stop]
                stored_supervised = gold_ids.ne(-100)
                safe_gold_ids = gold_ids.clamp_min(0)
                gold_logits = float_logits.gather(
                    dim=-1,
                    index=safe_gold_ids.unsqueeze(-1),
                ).squeeze(-1)
                gold_token_logits = torch.zeros_like(full_logsumexp)
                gold_token_nll = torch.zeros_like(full_logsumexp)
                gold_token_logits[stored_supervised] = gold_logits[stored_supervised]
                gold_token_nll[stored_supervised] = (
                    full_logsumexp[stored_supervised]
                    - gold_logits[stored_supervised]
                )

                token_metric = F.cross_entropy(
                    float_logits.contiguous().view(-1, logits.size(-1)),
                    gold_ids.contiguous().view(-1),
                    reduction="none",
                ).view(input_ids.size(0), -1)
                chunk_metric_mask = metric_mask[..., start:stop]
                metric_sum += (token_metric * chunk_metric_mask).sum(dim=-1)

                selected_logit_chunks.append(
                    selected_logits.detach().float().cpu()
                )
                selected_index_chunks.append(selected_indices.detach().cpu())
                full_logsumexp_chunks.append(full_logsumexp.detach().cpu())
                gold_token_logit_chunks.append(gold_token_logits.detach().cpu())
                gold_token_nll_chunks.append(gold_token_nll.detach().cpu())

                del (
                    cache_position,
                    chunk_metric_mask,
                    float_logits,
                    full_logsumexp,
                    gold_ids,
                    gold_logits,
                    gold_token_logits,
                    gold_token_nll,
                    logits,
                    output,
                    safe_gold_ids,
                    selected_indices,
                    selected_logits,
                    stored_supervised,
                    token_metric,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            metric = metric_sum / supervised_count
            if not torch.isfinite(metric).all():
                raise ValueError("CE produced a non-finite per-sample loss")

            inputs[PER_STEP_LOGITS] = torch.cat(
                selected_logit_chunks,
                dim=1,
            )
            inputs[PER_STEP_INDICES] = torch.cat(
                selected_index_chunks,
                dim=1,
            )
            inputs[FULL_LOGSUMEXP] = torch.cat(
                full_logsumexp_chunks,
                dim=1,
            )
            inputs[GOLD_TOKEN_IDS] = gold_token_ids.detach().cpu()
            inputs[GOLD_TOKEN_LOGITS] = torch.cat(
                gold_token_logit_chunks,
                dim=1,
            )
            inputs[GOLD_TOKEN_NLL] = torch.cat(
                gold_token_nll_chunks,
                dim=1,
            )
            inputs[METRIC] = metric.detach().cpu()

            del (
                full_logsumexp_chunks,
                gold_token_ids,
                gold_token_logit_chunks,
                gold_token_nll_chunks,
                metric,
                metric_mask,
                metric_sum,
                past_key_values,
                selected_index_chunks,
                selected_logit_chunks,
                supervised_count,
            )

    if was_training:
        model.train()
    del input_ids, attention_mask, labels
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return inputs
