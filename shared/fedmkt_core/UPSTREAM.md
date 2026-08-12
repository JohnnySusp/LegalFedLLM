# FedMKT upstream record

The optional machine-learning modules in this directory were extracted and adapted from:

- Repository: `FederatedAI/FATE-LLM`
- Commit: `0c63377e468f0f62a9bdf5fb32424688b9478553`
- Package: `fate_llm.algo.fedmkt`
- License: Apache License 2.0

The original FATE and FuseAI/FuseLLM notices remain in the adapted source files.
LegalFedLLM changes imports from the FATE package namespace to this local package,
uses `AutoTokenizer` instead of the FATE tokenizer factory, and makes logit extraction
device-neutral. It also replaces the upstream `editdistance.eval` calls with
`rapidfuzz.distance.Levenshtein.distance` so Python 3.14 installations can use a
prebuilt wheel. The logit-generation adaptation disables the model cache, stores
selected logits as float32, and normalizes CE over supervised non-padding labels
rather than every attended token. The latter preserves upstream whole-sequence CE
when all labels are supervised and correctly supports LegalFedLLM's signed
`chat_sft_answer_only_v1` label format. Operational teacher selection retains every
sample and chooses the first minimum-CE source from the Host followed by trusted
Clients in signed manifest order. The imported trainer applies distillation only
at causal positions whose next-token label is supervised. FATE Context,
Guest/Host/Arbiter channels, FATE-Flow and aggregation wrappers are not included.

The protocol-first selection and safety modules remain dependency-free. Install
`requirements.txt` before importing the upstream-derived `shared.fedmkt_core.ml`
modules.
