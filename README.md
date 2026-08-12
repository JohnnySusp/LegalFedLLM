# LegalFedLLM

LegalFedLLM is a protocol-first proof of concept for bidirectional, FedMKT-style
knowledge transfer between heterogeneous language models in legal environments.
Participants keep private examples and model-native LoRA tensors local. They
exchange signed Knowledge Packages containing outputs over a shared public
reference dataset instead of attempting to average structurally incompatible
adapters.

The repository currently combines:

1. a deterministic mock backend for fast protocol, persistence and security
   testing;
2. a real canonical shared-reference-dataset boundary;
3. a deterministic importer for the pinned 2012 Greek Law Digest thesis copy;
4. signed schema 2.0 Knowledge Packages with exact `safetensors` artifacts; and
5. one real Client path using a pinned Transformers model and PEFT LoRA.

Steps 0 through 3 are complete at the current Client boundary. The authoritative
Step 3 run trained a real Qwen LoRA candidate, performed teacher-forced inference
over all 565 accepted D^P samples, produced real top-k logits and answer-only
cross-entropy losses, created a signed schema 2.0 package and validated the
existing Coordinator intake path. Host-side token alignment and real selective
distillation begin in Step 4 and remain unimplemented.

The copyrighted GLD source PDF, generated datasets, private Client examples,
downloaded model files and trained adapters remain local and are excluded from
Git.

## Current implementation boundary

The real Client path is:

```text
signed round manifest
        ↓
verified, frozen D^P sample order
        ↓
private Client JSONL examples
        ↓
pinned base model and tokenizer
        ↓
answer-only PEFT LoRA training
        ↓
adapter-only checkpoint validation and atomic promotion
        ↓
teacher-forced inference over D^P
        ↓
raw top-k token IDs and logits + answer-only CE per sample
        ↓
deterministic safetensors Knowledge Artifact
        ↓
signed schema 2.0 Client Knowledge Package
        ↓
bounded multipart Coordinator intake and immutable persistence
```

The complete mock round continues beyond this point through deterministic
DualMinCE selection, Host candidate promotion or rollback, signed Host package
publication and Client reverse synchronization. Those mock stages prove the
protocol and service behavior, not real Host or reverse-Client learning.

No Client LoRA tensor is sent to the Coordinator or Host. Private prompt and
answer text is absent from the signed package metadata and numerical artifact.

## Implemented milestones

- **Step 0 — protocol-first baseline:** signed manifests and Knowledge Packages,
  bounded asynchronous rounds, filesystem persistence, replay protection,
  deterministic DualMinCE selection, Host validation and rollback, Client
  synchronization and an Ollama serving boundary.
- **Step 1A — shared dataset boundary:** canonical samples, JSONL I/O, semantic
  identity, deterministic D^P/D^V splitting, Coordinator snapshots, selected-
  Client delivery and independent Client/Host verification.
- **Step 1B — GLD importer:** deterministic extraction from the pinned source,
  reviewed follow-up handling, subsection disambiguation, text-hygiene checks,
  corpus auditing and reproducible D^P/D^V generation.
- **Step 2 — scalable signed packages:** schema 2.0 JSON envelopes, schema 1.0
  `safetensors` artifacts, exact descriptor binding, bounded multipart transport,
  immutable persistence, security regressions and complete mock-round use in both
  directions.
- **Step 3.1 — pinned real Client contracts:** exact model/tokenizer revisions,
  strict private-data schema, answer-only labels and manifest-bound training
  settings.
- **Step 3.2 — checkpoint lifecycle:** round-specific training records,
  adapter-only checkpoints, validation, atomic promotion and restart recovery.
- **Step 3.3 — real PEFT execution:** CUDA/BF16 Transformers training, fresh
  save/reload validation and deterministic probe-logit equivalence.
- **Step 3.4 — training hardening:** optimizer-step and finite-loss checks,
  changed-LoRA verification, optional frozen-base checksum, failure cleanup and
  one process-local ML lock.
- **Step 3.5 — real D^P knowledge generation:** exact signed sample order,
  overlength rejection, batched no-grad inference, raw top-k extraction and
  answer-only causal CE.
- **Step 3.6 — real package submission:** existing artifact writer, signature and
  multipart intake reused without a parallel real-only protocol; immutable retry
  and exact training/checkpoint provenance are enforced.

## Repository layout

```text
LegalFedLLM/
├── client/
│   ├── Dockerfile              Lightweight and ML image targets
│   ├── main.py                 Client HTTP API and shared ML lock
│   ├── runtime.py              Client state, package and round integration
│   ├── model_profiles.py       Exact pinned Client model profiles
│   ├── training.py             Data contracts and checkpoint lifecycle
│   ├── peft_backend.py         Real Transformers/PEFT execution
│   └── knowledge.py            D^P encoding and KnowledgeSample conversion
├── coordinator/
│   ├── main.py                 Public federation API
│   ├── service.py              Round orchestration and Host gateway
│   └── reference_data.py       Coordinator-owned D^P/D^V boundary
├── host/
│   ├── main.py                 Private internal Host API
│   └── runtime.py              Host state and mock distillation path
├── shared/
│   ├── protocol.py             Manifests, profiles and package schemas
│   ├── knowledge_artifact.py   Deterministic safetensors artifact I/O
│   ├── knowledge_transport.py  Bounded streaming multipart transport
│   ├── reference_dataset.py    Canonical schema, JSONL I/O and hashing
│   ├── prompt.py               Shared reference-prompt renderer
│   ├── crypto.py               Ed25519 and canonical SHA-256 helpers
│   ├── storage.py              Atomic cross-platform persistence
│   ├── ollama.py               Optional Ollama serving connector
│   ├── fedmkt_runtime.py       Mock/real FedMKT boundary
│   └── fedmkt_core/
│       ├── selection.py        Dependency-free DualMinCE selection
│       ├── safety.py           Protocol-first safety checks
│       ├── UPSTREAM.md         FATE-LLM extraction and adaptation record
│       └── ml/                 Adapted optional FedMKT components
├── tools/datasets/
│   ├── inspect_gld_layout.py   Offline PDF layout inspection
│   └── gld_pdf_to_jsonl.py     Deterministic pinned-GLD importer
├── tests/
│   ├── test_client_training.py
│   ├── test_client_knowledge.py
│   ├── test_client_real_package.py
│   ├── test_client_real_model.py
│   └── ...                     Dataset, package, transport and round tests
├── scripts/
│   ├── demo_round.py           Containerized mock-round driver
│   └── measure_knowledge_packages.py
├── .env.example
├── compose.yaml
├── requirements.txt
└── THIRD_PARTY_NOTICES.md
```

All Python dependencies are consolidated in `requirements.txt`. Heavy ML imports
remain lazy where practical so the ordinary test suite does not load a model or
require a GPU.

## Pinned real Client profile

The authoritative Step 3 profile is:

| Field | Value |
| --- | --- |
| Profile | `qwen3-1.7b-lora-v1` |
| Model/tokenizer | `Qwen/Qwen3-1.7B` |
| Revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Model class | `Qwen3ForCausalLM` |
| Tokenizer class | `Qwen2TokenizerFast` |
| Vocabulary | 151,936 tokens |
| Chat mode | Qwen non-thinking |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Precision | BF16 |
| Quantization | none |

The repository also defines `llama-3.2-1b-instruct-lora-v1` at exact revision
`9213176726f574b556790deb65791e0c5aa438b6`, but the accepted Step 3 run used Qwen.
An arbitrary Hugging Face or Ollama model is not automatically a supported
training profile.

Relevant dependency pins are:

```text
accelerate==1.14.0
peft==0.20.0
rapidfuzz==3.14.5
safetensors==0.8.0
sentencepiece==0.2.2
torch==2.13.0
transformers==4.57.6
PyMuPDF==1.28.0
```

The adapted FedMKT code uses RapidFuzz Levenshtein distance in place of
`editdistance` for Python 3.14 wheel compatibility. See
`shared/fedmkt_core/UPSTREAM.md` and `THIRD_PARTY_NOTICES.md`.

## Private Client training data

The real Client reads local UTF-8 JSONL. Each line has exactly this logical form:

```json
{
  "schema_version": "1.0",
  "example_id": "private-example-001",
  "prompt": "A private local instruction or question.",
  "answer": "The private local target answer."
}
```

Requirements include:

- unique non-empty example IDs;
- non-empty prompt and answer strings;
- one consistent supported schema;
- deterministic order and dataset hashing;
- no truncation of examples that exceed the signed sequence limit; and
- answer-only supervision under `chat_sft_answer_only_v1`.

Compose mounts the host directory configured by `CLIENT_PRIVATE_DATA_DIR` as
read-only `/private`, with the training file expected at `/private/train.jsonl`.
The examples and their text never enter Coordinator storage. Training records
persist only identity hashes, counts, settings and measured execution results.

## Round-bound training and checkpoints

For real training, the signed manifest fixes:

- the selected Client and exact model-profile hash;
- D^P identity and ordered sample IDs;
- prompt-template and label-format identity;
- maximum sequence length and `truncation_policy=reject`;
- training epochs and top-k; and
- DP-policy report fields.

The execution profile fixes the local device, precision, micro-batch size,
gradient accumulation, learning rate, seed, scheduler and optional frozen-base
checksum.

A candidate is promoted only when all of the following succeed:

- at least one optimizer step ran;
- training loss is finite and non-negative;
- at least one LoRA tensor differs from its parent;
- the checkpoint contains adapter files rather than full base-model weights;
- metadata, tensor shapes, ranks and target modules match the pinned profile;
- an optional full frozen-base checksum remains unchanged; and
- a fresh base-model-plus-adapter reload matches the in-memory probe logits at
  strict `atol=1e-4`.

Candidate and Trainer staging directories are removed after success or failure.
A failed run cannot advance the current-adapter pointer or create a completed
round record.

The Client uses one process-local ML lock for both real training and real knowledge
generation. Work runs outside FastAPI's event loop, `/health` stays responsive and
a concurrent training or generation request receives HTTP 409.

## Real D^P knowledge contract

Every reference sample is rendered with the same pinned chat and answer-only
label contract used for private training.

```text
input              rendered public prompt + gold answer + tokenizer terminator
stored positions   every non-padding source position
top-k              highest raw model logits at each stored position
sample metric      causal CE averaged over supervised answer targets only
padding            allowed for batches, removed from stored samples
inference          eval mode, no gradients, use_cache=False
order              exact signed D^P sample order
overlength         reject; never truncate
```

The package generator loads the archived round-specific adapter named by the
validated training record, not whichever adapter happens to be current later.
Pending retries reuse the exact signed package and exact artifact bytes without a
second inference run.

## Reference dataset boundary

A canonical reference sample contains:

```json
{
  "schema_version": 1,
  "dataset_id": "example-reference",
  "dataset_version": "v1",
  "sample_id": "example-ch001-s001-q001",
  "chapter": "Example Chapter",
  "section": "Example Section",
  "question": "What is the question?",
  "gold_answer": "The original gold answer.",
  "source": {
    "document_id": "example-document",
    "page_start": 10,
    "page_end": 11
  }
}
```

The authoritative runtime format is UTF-8 JSONL with one sample per line. Pretty
JSON files are generated only for human inspection.

The semantic dataset hash covers the schema, dataset ID/version and ordered
`sample_id`, `chapter`, `section`, `question` and `gold_answer` fields. Source page
metadata is provenance and is excluded from the semantic hash.

The generic split groups samples by `(chapter, section)` and preserves source
order. A one-sample section belongs entirely to D^P. Otherwise D^P receives the
first `floor(0.8 * n)` samples and D^V receives the remainder. GLD-dependent
follow-up pairs are grouped by the source-specific importer before this generic
split, so a base question and its dependent follow-up cannot be separated.

The shared prompt renderer produces:

```text
Chapter: {chapter}

Section: {section}

Question: {question}

Answer:
```

`gold_answer` remains a separate target and is not inserted into this public
prompt string.

### Accepted GLD corpus identity

The authoritative importer run used PyMuPDF 1.28.0 and the pinned 713-page Greek
Law Digest source.

| Dataset | Samples | Semantic SHA-256 |
| --- | ---: | --- |
| Complete corpus | 738 | `cf5c81dcecaab58848c1afb0e99f86bcf5fd32823c2aaee34a65f6f4a2ccd0e8` |
| D^P reference corpus | 565 | `5d855a429d43b70eb146aeb11cda1f675c05d6465bea0792796fdcd8d6ceb231` |
| D^V validation corpus | 173 | `1e40a74799b9900ff8b9a9e05dd379fd0c00226370625f7da1fdca13142b83b5` |

The accepted audit reported zero unresolved follow-ups, duplicate canonical
prompts, contributing-firm/profile contamination, out-of-scope samples and
blocking issues.

## Knowledge Package and Artifact contract

Knowledge Package schema 2.0 is one signed logical object with two transported
parts:

```text
canonical signed JSON envelope
        +
exact safetensors numerical artifact
```

The envelope's descriptor binds the exact artifact bytes:

```json
{
  "format": "safetensors",
  "schema_version": "1.0",
  "byte_size": 4809632,
  "sha256": "<SHA-256 of exact artifact bytes>",
  "sample_count": 565,
  "sample_ids_sha256": "<SHA-256 of the canonical ordered ID list>",
  "total_token_count": 133336,
  "top_k": 4
}
```

The byte size and token count above are from the accepted real Step 3 package;
run-specific hashes are deliberately not frozen as project constants.

Artifact schema 1.0 contains exactly six tensors. `N` is the sample count, `T`
the total stored token count and `K` top-k.

| Tensor | Dtype | Shape | Purpose |
| --- | --- | --- | --- |
| `sample_offsets` | `int64` | `[N + 1]` | Variable-length sample boundaries |
| `source_input_ids` | `int32` | `[T]` | Sender-tokenizer input IDs |
| `attention_lengths` | `int32` | `[N]` | Unmasked length per sample |
| `top_k_token_ids` | `int32` | `[T, K]` | Sender-tokenizer top-k IDs |
| `top_k_logits` | `float32` | `[T, K]` | Raw top-k logits |
| `ce_losses` | `float32` | `[N]` | Per-sample cross-entropy loss |

The verification chain is:

```text
Ed25519 signature
    → signed package metadata and package hash
    → artifact descriptor
    → exact artifact byte size and SHA-256
    → strict tensor, shape, dtype, order and finite-number validation
```

Multipart upload/download is bounded and streaming. Rejected or interrupted
transfers are cleaned up. Accepted Coordinator submissions are immutable:

```text
rounds/<round-id>/submissions/<client-id>/package.json
rounds/<round-id>/submissions/<client-id>/knowledge.safetensors
```

### Step 2 representation measurement

The retained measurement utility compares schema 2.0 metadata plus the exact
artifact with the former embedded-JSON representation:

```bash
python scripts/measure_knowledge_packages.py \
  --reference data/derived/gld2012/reference.jsonl
```

It refuses a reference dataset whose count or semantic hash differs from the
accepted 565-sample D^P. The accepted deterministic mock run used top-k 20:

| Representation | Client | Host |
| --- | ---: | ---: |
| Canonical package JSON | 16,246 B | 16,234 B |
| `safetensors` artifact | 1,121,448 B | 1,121,448 B |
| Logical package | 1,137,694 B | 1,137,682 B |
| Multipart body | 1,138,040 B | 1,138,028 B |
| Equivalent embedded JSON | 2,159,947 B | 2,160,046 B |
| Size reduction | 47.3277% | 47.3307% |

These are representation measurements of deterministic mock knowledge, not
model-quality, training-time, memory or privacy results.

## Step 3 authoritative acceptance

The accepted environment reported:

```text
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
PyTorch: 2.13.0+cu130
CUDA runtime: 13.0
CUDA available: true
BF16 supported: true
model profile: qwen3-1.7b-lora-v1
model revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```

The frozen D^P exposed 37 samples longer than 512 Qwen tokens. The largest was
1,814 tokens, so the accepted signed sequence limit was 2,048 with rejection
rather than truncation.

The full real package reported:

| Measurement | Observed value |
| --- | ---: |
| D^P samples | 565 |
| Stored source tokens | 133,336 |
| Artifact size | 4,809,632 bytes |
| Minimum answer-only CE | 1.5879086256027222 |
| Maximum answer-only CE | 17.829463958740234 |
| Trainable LoRA parameters | 3,211,264 |
| Total model parameters | 1,723,786,240 |
| Optimizer steps | 1 |
| Frozen-base checksum | verified unchanged |

The artifact is below the existing 25 MiB logical package limit. Package,
artifact and checkpoint hashes were produced and validated for each isolated run,
but they are run-specific rather than frozen protocol constants.

Peak RAM, peak VRAM and wall-clock timings were not recorded as stable thesis
measurements in Step 3 and are not claimed here. Comparative resource and timing
measurement belongs to the later complete heterogeneous-round experiment.

## Test and verification commands

### Lightweight repository suite

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
```

The authoritative Step 3 run reported:

```text
Ran 133 tests
OK (skipped=2)
```

The two skips are the opt-in real-model acceptance tests. The ordinary suite
does not download or load a language model.

Focused model-free Client tests can be run with:

```bash
python -m unittest -v \
  tests.test_client_training \
  tests.test_client_knowledge \
  tests.test_client_real_package
```

### Build and health-check the services

Create an environment file and replace all development tokens:

```bash
cp .env.example .env
mkdir -p data/client-private
```

For ordinary service startup, provide `data/client-private/train.jsonl` in the
format documented above, then run:

```bash
docker compose -p legalfedllm up --build --wait --wait-timeout 180
docker compose -p legalfedllm ps
```

Host, Coordinator and Client should all report `healthy`. The Compose Client uses
the `client-ml` target, the Qwen profile, all available GPUs, 2 GiB shared memory,
a persistent Hugging Face cache and a read-only private-data mount.

Check the published services:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8001/health
```

Reset prototype state only when that destructive cleanup is intended:

```bash
docker compose -p legalfedllm down -v
```

### Focused real-model acceptance

This downloads the pinned model if it is not already cached and requires a
compatible CUDA/BF16 environment:

```bash
docker compose -p legalfedllm run --rm --no-deps -T \
  -e LEGALFEDLLM_RUN_REAL_MODEL_TESTS=true \
  -e LEGALFEDLLM_REAL_MODEL_PROFILE=qwen3-1.7b-lora-v1 \
  client python -m unittest -v \
  tests.test_client_real_model.RealClientModelAcceptanceTests.test_one_peft_step_save_reload_and_real_knowledge
```

The acceptance test creates two temporary private examples, performs one real
LoRA optimization, validates the checkpoint, creates a two-sample real package,
reloads it and verifies byte-identical retry behavior.

### Full frozen-D^P real package

The accepted GLD D^P must exist locally at
`data/derived/gld2012/reference.jsonl`:

```bash
docker compose -p legalfedllm run --rm --no-deps -T \
  -v "$PWD/data/derived/gld2012:/datasets:ro" \
  -e LEGALFEDLLM_RUN_REAL_MODEL_TESTS=true \
  -e LEGALFEDLLM_REAL_MODEL_PROFILE=qwen3-1.7b-lora-v1 \
  -e LEGALFEDLLM_REAL_REFERENCE_DATASET_PATH=/datasets/reference.jsonl \
  -e LEGALFEDLLM_REAL_MAX_SEQUENCE_LENGTH=2048 \
  client python -m unittest -v \
  tests.test_client_real_model.RealClientModelAcceptanceTests.test_full_565_sample_reference_dataset
```

The test refuses the wrong dataset identity/order, rejects overlength input before
expensive inference and validates the generated signed package and artifact.

## Offline GLD dataset tooling

Place the pinned source at:

```text
data/private/greek_law_digest.pdf
```

Then run:

```bash
python tools/datasets/inspect_gld_layout.py
python tools/datasets/gld_pdf_to_jsonl.py
```

The importer accepts only the pinned source:

```text
expected PDF pages: 713
expected SHA-256:
9673ee7c86b3d582e2c08e1cdd2b84f144981f31a1fe50d4216e82c5b350b77d
```

Its fixed initial scope begins at printed page 34 and ends before `COVERED BONDS`
on printed page 306. It imports 44 Q&A-structured sections and explicitly excludes
two prose-only sections rather than synthesizing questions.

A clean run writes:

```text
data/derived/gld2012/
├── all.jsonl
├── all.json
├── reference.jsonl
├── reference.json
├── validation.jsonl
├── validation.json
├── identity.json
└── review.json
```

If unresolved candidates or audit failures remain, final corpus files are removed
and only candidate/review outputs are retained. Generated data must be regenerated
after importer changes rather than edited manually.

## Service APIs

### Coordinator — published port 8000

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status |
| `GET` | `/v1/identity` | Coordinator and Host public identity |
| `POST` | `/v1/clients/register` | Register a Client profile and public key |
| `POST` | `/v1/rounds` | Create and sign a round manifest |
| `GET` | `/v1/rounds/current` | Retrieve the current manifest |
| `GET` | `/v1/rounds/{id}/manifest` | Retrieve one manifest |
| `GET` | `/v1/rounds/{id}/reference-dataset` | Download selected round D^P |
| `POST` | `/v1/rounds/{id}/knowledge` | Upload package JSON plus artifact |
| `GET` | `/v1/rounds/{id}/status` | Poll round state |
| `GET` | `/v1/rounds/{id}/host-knowledge` | Download signed Host package |
| `POST` | `/v1/generate` | Proxy direct Host consultation |

### Client — loopback-published port 8001

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Client state and backend status |
| `POST` | `/v1/register` | Register with the Coordinator |
| `POST` | `/v1/local-train` | Local training outside a round |
| `POST` | `/v1/rounds/{id}/local-train` | Train against one signed round |
| `POST` | `/v1/participate` | Legacy current-round participation |
| `POST` | `/v1/rounds/{id}/participate` | Generate and submit the round package |
| `POST` | `/v1/rounds/{id}/sync` | Consume verified Host knowledge |
| `POST` | `/v1/generate` | Local mock or Ollama inference |
| `GET` | `/v1/ollama/models` | List installed Ollama models |
| `POST` | `/v1/ollama/inspect` | Inspect one Ollama model |

Client administrative endpoints require `X-Client-Admin-Token`. `/health` and
`/v1/generate` remain outside that administrative gate.

### Host — private port 8002

The Host API is reachable only inside the Compose network and is protected by
`X-Internal-Token`. It owns dataset verification, current Host adapter metadata,
mock selection/distillation, validation and rollback, signed Host publication and
optional Ollama inference.

## Persistence and retry behavior

No database is required. Each service owns a local filesystem state tree.

Client state distinguishes:

```text
identity and active state
round-specific training records
versioned PEFT adapter checkpoints
verified D^P caches
pending Knowledge Packages
accepted Knowledge Packages
Host package caches
submission receipts
pending and accepted adapter snapshots
```

For a real package, the adapter version and checkpoint hash come from the signed-
round training record, not mutable current state. Pending and accepted package
state is revalidated against the round, model profile, training record, checkpoint,
package signature and artifact before reuse.

A retry while pending reuses the exact JSON and artifact bytes. After Coordinator
acceptance, the local package/artifact/snapshot set is immutable. A second accepted
submission is rejected rather than processed twice.

## Security included now

The current implementation includes:

- Ed25519 identities and signatures;
- canonical JSON hashing and signing;
- exact artifact size and SHA-256 binding;
- registered Client public keys and signed Coordinator manifests;
- round, manifest, dataset, sample-order, model, tokenizer, adapter and DP-report
  binding;
- persistent nonce and package-hash replay protection;
- selected-Client authorization for D^P download;
- bounded logical-package and multipart sizes;
- strict artifact names, dtypes, shapes, offsets and finite-number checks;
- temporary-file cleanup and immutable accepted storage;
- round-specific training/checkpoint provenance;
- a minimal deterministic Knowledge Package safety gate; and
- append-only JSONL audit events.

Ed25519 proves origin and integrity; it does not encrypt traffic. Compose uses
plain HTTP on its development network. Registration, administrative and internal
tokens are development credentials rather than production identities.

The current DP report checks policy and protocol consistency. Real training is
ordinary LoRA training, not DP-SGD, and no formal privacy accountant is claimed.

## Ollama boundary

Ollama remains an optional serving boundary. It is not the real training runtime.

```env
CLIENT_SERVING_BACKEND=ollama
CLIENT_OLLAMA_MODEL=<installed model name>
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

For Bazzite/Podman with host Ollama, use:

```env
OLLAMA_BASE_URL=http://host.containers.internal:11434
```

LegalFedLLM trains through Transformers/PEFT. Automatic export of a promoted PEFT
adapter into an Ollama model profile is future work.

## Extracted FedMKT core

Optional ML modules under `shared/fedmkt_core/ml/` were extracted and adapted
from FATE-LLM commit:

```text
0c63377e468f0f62a9bdf5fb32424688b9478553
```

Included components cover top-k/CE generation, token alignment, vocabulary
mapping, `DataCollatorForFedMKT`, `FedMKTTrainer` and constants. FATE Context,
FATE communication roles, FATE-Flow and aggregation wrappers are not included.
LegalFedLLM supplies the HTTP, security, persistence and round layers.

## Current limitations

LegalFedLLM does not yet provide:

- real Client-to-Host or Host-to-Client token/vocabulary alignment;
- real Host baseline inference or selective LoRA distillation;
- real D^V validation and Host adapter promotion/rollback;
- real reverse Client distillation from a Host package;
- a complete heterogeneous-model federation round;
- DP-SGD or formal differential-privacy accounting;
- a learned malicious-package detector;
- HTTPS, production identity bootstrap or encrypted artifact storage;
- automatic PEFT-to-Ollama adapter publication; or
- a graphical application.

The deterministic mock backend must remain available while these real stages are
added.

## Next implementation milestones

### Step 4 — FedMKT parity and token alignment

Verify the extracted FedMKT behavior before connecting it to a live Host path:

```text
Client-to-Host token alignment
Host-to-Client token alignment
vocabulary mapping
DataCollatorForFedMKT behavior
FedMKTTrainer behavior
DualMinCE parity
```

### Step 5 — real Host distillation

Add real Host baseline inference, aligned selective knowledge distillation, D^V
validation and adapter promotion or rollback.

### Step 6 — real reverse Client distillation

Verify and align Host knowledge, select samples where the Host teacher is better
and distil them into a Client-specific LoRA.

### Step 7 — complete heterogeneous-model round

Run multiple real Clients against a heterogeneous Host and measure wall time,
peak VRAM/RAM, communication volume, selected teaching samples, validation
behavior, adapter sizes and rollback behavior.

## Accurate project claim

LegalFedLLM now demonstrates a real, pinned Client model that can learn a local
LoRA adapter from private examples and turn its outputs over the complete frozen
565-sample D^P into a signed, validated and transportable Knowledge Package.
Private examples and Client-native LoRA tensors remain local.

It does **not** yet demonstrate real heterogeneous Host learning, reverse real
Client learning, a completed real-model federation round, formal differential
privacy, a complete Safe-FedLLM defense or production-ready deployment.
