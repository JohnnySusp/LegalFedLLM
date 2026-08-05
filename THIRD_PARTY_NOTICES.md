# Third-party notices

The optional FedMKT machine-learning core under `shared/fedmkt_core/ml/` was
extracted and adapted from FederatedAI/FATE-LLM commit
`0c63377e468f0f62a9bdf5fb32424688b9478553` under the Apache License 2.0.
The FATE and FuseAI/FuseLLM notices present in the source files are retained.
See `shared/fedmkt_core/LICENSE` and `shared/fedmkt_core/UPSTREAM.md`.

## Optional dataset tooling

The offline dataset-preparation utilities under `tools/datasets/` may use
PyMuPDF 1.28.0. PyMuPDF is dual-licensed under the GNU Affero General
Public License v3.0 or an Artifex commercial license.

PyMuPDF is not part of the LegalFedLLM runtime dependency set. It is used
only by optional, source-specific offline dataset-preparation utilities.