from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from shared.fedmkt_core import dual_min_ce_select, inspect_knowledge_package
from shared.protocol import KnowledgeSample, RoundManifest


def _seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def deterministic_knowledge_samples(
    *,
    manifest: RoundManifest,
    participant_id: str,
    role: str,
    adapter_version: int,
) -> list[KnowledgeSample]:
    samples: list[KnowledgeSample] = []
    sequence_length = min(12, manifest.maximum_sequence_length)
    vocabulary_size = 32_000

    for sample_id in manifest.sample_ids:
        input_rng = random.Random(_seed(manifest.round_id, sample_id, "reference-input"))
        source_input_ids = [input_rng.randrange(vocabulary_size) for _ in range(sequence_length)]
        rng = random.Random(
            _seed(manifest.round_id, sample_id, participant_id, role, str(adapter_version))
        )
        token_rows: list[list[int]] = []
        logit_rows: list[list[float]] = []
        for _ in range(sequence_length):
            tokens = rng.sample(range(vocabulary_size), manifest.top_k)
            logits = sorted(
                [round(rng.uniform(-2.0, 8.0), 6) for _ in range(manifest.top_k)],
                reverse=True,
            )
            token_rows.append(tokens)
            logit_rows.append(logits)

        if role == "host":
            base = 0.80 + rng.random() * 0.10
            ce_loss = max(0.01, base - adapter_version * 0.50)
        else:
            ce_loss = 0.30 + rng.random() * 0.40

        samples.append(
            KnowledgeSample(
                sample_id=sample_id,
                source_input_ids=source_input_ids,
                attention_length=sequence_length,
                top_k_token_ids=token_rows,
                top_k_logits=logit_rows,
                ce_loss=round(ce_loss, 8),
            )
        )
    return samples


@dataclass(frozen=True)
class FedMKTCore:
    dual_min_ce_select = staticmethod(dual_min_ce_select)
    inspect_knowledge_package = staticmethod(inspect_knowledge_package)


def load_ml_components():
    from shared.fedmkt_core.ml import (
        DataCollatorForFedMKT,
        FedMKTTrainer,
        generate_pub_data_logits,
        token_align,
    )

    return {
        "trainer": FedMKTTrainer,
        "collator": DataCollatorForFedMKT,
        "generate_knowledge": generate_pub_data_logits,
        "token_align": token_align,
    }
