# Third-party notices

The optional FedMKT machine-learning core under `shared/fedmkt_core/ml/` was
extracted and adapted from FederatedAI/FATE-LLM commit
`0c63377e468f0f62a9bdf5fb32424688b9478553` under the Apache License 2.0.
The FATE and FuseAI/FuseLLM notices present in the source files are retained.
See `shared/fedmkt_core/LICENSE` and `shared/fedmkt_core/UPSTREAM.md`.

## Knowledge Artifact runtime

The scalable Knowledge Artifact boundary uses NumPy under the BSD 3-Clause
License and `safetensors` under the Apache License 2.0.

Multipart Knowledge Package transport uses `python-multipart` under the
Apache License 2.0.

## Optional model training

The optional real-model path uses PyTorch, Transformers, PEFT, Accelerate,
SentencePiece and RapidFuzz. They are pinned with the other project dependencies
in `requirements.txt`. The mock path does not import the optional model-training
modules. Consult each dependency's distributed licence and notice files.

The supported Qwen proof-of-concept profile identifies
[`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B). Qwen3 is made available under
the Apache License 2.0. LegalFedLLM does not bundle its weights.

The Host proof-of-concept profile identifies
[`ibm-granite/granite-3.3-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct).
Granite 3.3 is made available under the Apache License 2.0. LegalFedLLM does not
bundle Granite weights, trained adapters or merged Ollama models.

## Optional dataset tooling

The offline dataset-preparation utilities under `tools/datasets/` may use
PyMuPDF 1.28.0. PyMuPDF is dual-licensed under the GNU Affero General
Public License v3.0 or an Artifex commercial license.

PyMuPDF is not part of the LegalFedLLM runtime dependency set. It is used
only by optional, source-specific offline dataset-preparation utilities.

## Optional desktop application and packaging

The desktop Client uses PySide6/Qt for Python. PySide6 is distributed under the
LGPLv3/GPLv3/commercial Qt licensing options; release builders must preserve the
applicable Qt/PySide notices and license terms.

PyInstaller is used as a build-time bundler for per-platform portable binaries.
It is not a cross-compiler, so Windows and Linux release artifacts are produced
on their respective operating systems. Consult the PyInstaller distribution for
its GPL-with-exception licensing terms and bundled bootloader notices.
