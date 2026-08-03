"""Optional FedMKT machine-learning components.

Import concrete modules only after installing requirements-ml.txt.
"""

__all__ = [
    "DataCollatorForFedMKT",
    "FedMKTTrainer",
    "generate_pub_data_logits",
    "token_align",
]


def __getattr__(name):
    if name == "DataCollatorForFedMKT":
        from shared.fedmkt_core.ml.data_collator import DataCollatorForFedMKT

        return DataCollatorForFedMKT
    if name == "FedMKTTrainer":
        from shared.fedmkt_core.ml.trainer import FedMKTTrainer

        return FedMKTTrainer
    if name == "generate_pub_data_logits":
        from shared.fedmkt_core.ml.logit_generation import generate_pub_data_logits

        return generate_pub_data_logits
    if name == "token_align":
        from shared.fedmkt_core.ml.token_alignment import token_align

        return token_align
    raise AttributeError(name)
