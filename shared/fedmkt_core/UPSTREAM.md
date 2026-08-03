# FedMKT upstream record

The optional machine-learning modules in this directory were extracted and adapted from:

- Repository: `FederatedAI/FATE-LLM`
- Commit: `0c63377e468f0f62a9bdf5fb32424688b9478553`
- Package: `fate_llm.algo.fedmkt`
- License: Apache License 2.0

The original FATE and FuseAI/FuseLLM notices remain in the adapted source files.
LegalFedLLM changes imports from the FATE package namespace to this local package,
uses `AutoTokenizer` instead of the FATE tokenizer factory, and makes logit extraction
device-neutral. FATE Context, Guest/Host/Arbiter channels, FATE-Flow and aggregation
wrappers are not included.

The protocol-first milestone imports only the dependency-free selection and safety
modules. Install `requirements-ml.txt` before importing `shared.fedmkt_core.ml`.
