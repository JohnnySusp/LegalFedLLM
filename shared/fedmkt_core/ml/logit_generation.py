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

from shared.fedmkt_core.ml.vars_define import METRIC, PER_STEP_INDICES, PER_STEP_LOGITS


class Metric:
    @classmethod
    def cal_metric(cls, logits, input_ids, attention_mask, labels, training_args):
        if training_args.metric_type == "ce":
            return cls.cal_ce(logits, input_ids, attention_mask, labels, training_args)
        raise NotImplementedError(f"metric={training_args.metric_type} is not implemented")

    @classmethod
    def cal_ce(cls, logits, input_ids, attention_mask, labels, training_args):
        metric = F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[..., 1:].contiguous().view(-1),
            reduction="none",
        ).view(logits.size(0), -1)
        mask = attention_mask[..., 1:]
        return (metric * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)


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


def generate_pub_data_logits(inputs, model, training_args, data_collator):
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

    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        metric = Metric.cal_metric(
            logits, input_ids, attention_mask, labels, training_args
        )
        if training_args.top_k_logits_keep is None:
            raise ValueError("top_k_logits_keep must be configured")
        selected_logits, selected_indices = LogitsSelection.select_logits(
            logits, training_args
        )
        inputs[PER_STEP_LOGITS] = selected_logits.detach().cpu()
        inputs[PER_STEP_INDICES] = selected_indices.detach().cpu()
        inputs[METRIC] = metric.detach().cpu()

    if was_training:
        model.train()
    del logits, input_ids, attention_mask, labels
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return inputs
