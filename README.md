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
5. one real Qwen Client path using Transformers and PEFT LoRA;
6. an operational CPU path from heterogeneous signed Client packages through
   exact tokenizer validation, demand-driven vocabulary mapping, DTW alignment,
   binary trust eligibility, minimum-CE selection and sparse trainer targets;
   and
7. a real Granite Host path for baseline inference, selective LoRA candidate
   training, private D^V validation, atomic promotion or rejection, signed
   post-decision publication and Coordinator round finalization; and
8. a real Qwen reverse-distillation path with immutable sparse jobs, archived-
   parent LoRA training, public held-out validation, a hash-bound SafeFed-style
   LoRA probe gate and compare-and-swap candidate adoption.

The accepted real-Client run trained a Qwen LoRA candidate, performed teacher-
forced inference over all 565 accepted D^P samples, produced real top-k logits
and answer-only cross-entropy losses, created a signed schema 2.0 package and
validated the existing Coordinator intake path. The alignment layer additionally
supports Qwen and Granite Client identities against a temporary Granite validation
Host. The repository includes a full-D^P acceptance runner; its authoritative
Bazzite report remains local under the ignored `artifacts/` boundary.

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

For the pinned Qwen-to-Granite profile, the real Host path continues through:

```text
verified Client and Host Knowledge Packages
        ↓
trust-gated DTW alignment and minimum-CE teacher selection
        ↓
immutable bounded sparse Host training artifact
        ↓
selective Granite LoRA candidate training and reload verification
        ↓
private active-versus-candidate evaluation over D^V
        ↓
atomic promotion or candidate discard with an immutable decision audit
        ↓
post-decision Host inference over D^P
        ↓
signed Host Knowledge Package verification and round completion
        ↓
Client-owned Host-to-Client alignment and strict lower-CE selection
        ↓
immutable 90% D^P reverse-training job
        ↓
one-epoch Qwen LoRA candidate training and reload verification
        ↓
independent 10% D^P quality and SafeFed-style safety gates
        ↓
atomic promotion, rejection or stale-parent discard
```

The complete mock path remains available, including deterministic Client reverse
training decisions. Mock validation and safety values are protocol fixtures, not
evidence about a real model or classifier.

No Client LoRA tensor is sent to the Coordinator or Host. Private prompt and
answer text is absent from the signed package metadata and numerical artifact.

## Implemented capabilities

- **Protocol-first baseline:** signed manifests and Knowledge Packages,
  bounded asynchronous rounds, filesystem persistence, replay protection,
  deterministic DualMinCE selection, Host validation and rollback, Client
  synchronization and an Ollama serving boundary.
- **Shared dataset boundary:** canonical samples, JSONL I/O, semantic
  identity, deterministic D^P/D^V splitting, Coordinator snapshots, selected-
  Client delivery and independent Client/Host verification.
- **GLD importer:** deterministic extraction from the pinned source,
  reviewed follow-up handling, subsection disambiguation, text-hygiene checks,
  corpus auditing and reproducible D^P/D^V generation.
- **Scalable signed packages:** schema 2.0 JSON envelopes, schema 1.0
  `safetensors` artifacts, exact descriptor binding, bounded multipart transport,
  immutable persistence, security regressions and complete mock-round use in both
  directions.
- **Pinned real Client contracts:** exact model/tokenizer revisions,
  strict private-data schema, answer-only labels and manifest-bound training
  settings.
- **Checkpoint lifecycle:** round-specific training records,
  adapter-only checkpoints, validation, atomic promotion and restart recovery.
- **Real PEFT execution:** CUDA/BF16 Transformers training, fresh
  save/reload validation and deterministic probe-logit equivalence.
- **Training hardening:** optimizer-step and finite-loss checks,
  changed-LoRA verification, optional frozen-base checksum, failure cleanup and
  one process-local ML lock.
- **Real D^P knowledge generation:** exact signed sample order,
  overlength rejection, batched no-grad inference, raw top-k extraction and
  answer-only causal CE.
- **Real package submission:** existing artifact writer, signature and
  multipart intake reused without a parallel real-only protocol; immutable retry
  and exact training/checkpoint provenance are enforced.
- **FedMKT alignment parity:** the approved DTW path, tie behavior,
  bidirectional mappings, cumulative cost matrix and sparse logit transformation
  are protected by fixed golden tests.
- **Pinned heterogeneous profiles:** exact Qwen and Granite Client identities,
  the temporary Granite validation Host, immutable tokenizer artifacts and two
  fail-closed Client-to-Host alignment contracts.
- **Operational sparse distillation inputs:** persistent hashed vocabulary maps,
  per-Client DTW alignment, trust/quorum enforcement, deterministic minimum-CE
  selection, answer-only labels and bounded CPU float32 sparse targets.
- **Real Granite Host lifecycle:** exact pinned baseline inference, immutable
  sparse training jobs, answer-only `0.9` supervised plus `0.1` selected-teacher
  optimization, changed-LoRA and frozen-base checks, candidate reload
  equivalence and restart-safe candidate storage.
- **Private Host validation and finalization:** macro mean answer-token CE over
  D^V controls the `0.001` promotion threshold; token-weighted CE is retained as
  a diagnostic, per-sample D^V metrics remain Host-local, rejected candidate
  weights are discarded, and the accepted adapter produces a signed D^P Host
  package before the Coordinator completes the round.
- **Immutable Client reverse jobs:** deterministic 90/10 D^P partitioning,
  Client-owned Host-to-Client alignment, strict Host-better selection, retained
  Client self-targets, one public-data epoch and exact parent/package/artifact
  hash binding.
- **Real Qwen reverse candidate lifecycle:** archived-parent loading,
  answer-only `0.9` supervised plus `0.1` sparse distillation loss, LoRA-only
  optimization, frozen-base verification, safetensors save/reload equivalence
  and restart-safe candidate storage.
- **Independent Client adoption gates:** macro held-out answer-token CE may
  regress by at most `0.001`; token-weighted CE and teacher-forced exact-match/
  ROUGE-L remain diagnostics; a Qwen-specific SafeFed-style probe must report
  maliciousness below `0.8`; stale parents and either failed gate discard the
  candidate without overwriting a newer adapter.
- **Full-corpus validation runner:** exact D^P identity enforcement, signed
  deterministic package round-trips, mixed-client alignment, two-pass
  determinism checks and a machine-readable local resource report.

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
│   ├── reverse_training.py     Reverse candidate, validation and decision flow
│   ├── safety_probe.py         Strict SafeFed-style linear-probe artifact gate
│   └── knowledge.py            D^P encoding and KnowledgeSample conversion
├── coordinator/
│   ├── main.py                 Public federation API
│   ├── service.py              Round orchestration and Host gateway
│   └── reference_data.py       Coordinator-owned D^P/D^V boundary
├── host/
│   ├── main.py                 Private internal Host API
│   ├── runtime.py              Host state, validation and publication flow
│   ├── model_profiles.py       Exact pinned Granite Host profile
│   ├── training.py             Host execution and validation contracts
│   └── peft_backend.py         Real Granite LoRA and inference execution
├── shared/
│   ├── alignment_profiles.py   Approved per-Client alignment contracts
│   ├── protocol.py             Manifests, profiles and package schemas
│   ├── knowledge_artifact.py   Deterministic safetensors artifact I/O
│   ├── knowledge_transport.py  Bounded streaming multipart transport
│   ├── client_reverse_artifact.py  Bounded reverse-training artifact I/O
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
│   ├── test_client_real_reverse.py
│   ├── test_client_reverse_training.py
│   └── ...                     Dataset, package, transport and round tests
├── scripts/
│   ├── demo_round.py           Containerized mock-round driver
│   ├── measure_knowledge_packages.py
│   └── validate_fedmkt_alignment.py  Full-D^P alignment acceptance runner
├── .env.example
├── compose.yaml
├── requirements.txt
└── THIRD_PARTY_NOTICES.md
```

All Python dependencies are consolidated in `requirements.txt`. Heavy ML imports
remain lazy where practical so the ordinary test suite does not load a model or
require a GPU.

## Pinned Client profiles

The real training and package-generation acceptance profile is:

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

Granite 3.3 2B is also pinned as a Client identity for heterogeneous alignment
validation:

| Field | Value |
| --- | --- |
| Profile | `granite-3.3-2b-instruct-client-lora-v1` |
| Model/tokenizer | `ibm-granite/granite-3.3-2b-instruct` |
| Revision | `652c333dc5066f2a1764854a1bcd0ce67163d74f` |
| Model class/type | `GraniteForCausalLM` / `granite` |
| Tokenizer class | `GPT2TokenizerFast` |
| Vocabulary | 49,159 tokens |
| Chat mode | Standard Granite instruct template |
| Ollama base | `granite3.3:2b` |

The Granite Client profile is approved for the deterministic alignment runner;
it has not yet completed a real private-data LoRA training and package-generation
acceptance run. An arbitrary Hugging Face or Ollama model is not automatically a
supported training or alignment profile.

## Temporary validation Host and alignment profiles

The temporary alignment-validation Host is:

| Field | Value |
| --- | --- |
| Profile | `granite-3.3-2b-instruct-host-lora-v1` |
| Model/tokenizer | `ibm-granite/granite-3.3-2b-instruct` |
| Revision | `652c333dc5066f2a1764854a1bcd0ce67163d74f` |
| Model class/type | `GraniteForCausalLM` / `granite` |
| Tokenizer class | `GPT2TokenizerFast` |
| Vocabulary | 49,159 tokens |
| Chat mode | Standard Granite instruct template |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Ollama base | `granite3.3:2b` |

The approved signed pairs are:

| Client | Alignment identity | Purpose |
| --- | --- | --- |
| Qwen 3 1.7B | `dtw:qwen3-1.7b--granite3.3-2b-v1` | Heterogeneous Qwen-to-Granite mapping and DTW |
| Granite 3.3 2B | `dtw:granite3.3-2b-client--granite3.3-2b-host-v1` | Separate Client/Host identities over the same tokenizer |

The public Granite checkpoint is licensed under Apache 2.0 and does not require
gated model access. Its Host role is provisional and validates the transfer
mechanism; it is not a production Host selection. Choosing another Host requires
new signed Qwen-to-Host and Granite-to-Host profiles and another full acceptance
run. An adapter for one base model or size is not compatible with another.

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

## Client reverse distillation and adoption

Client synchronization first verifies the signed post-decision Host package and
reuses the accepted Client package, adapter snapshot and pinned Host-to-Client
alignment profile. D^P is deterministically divided into an approximately 90%
transfer subset and a 10% Client-validation subset. Both subsets retain signed
order, and their semantic hashes are bound into the reverse job.

Every transfer sample remains in the training artifact. The Host supplies the
sparse target only when its answer-token mean CE is strictly lower; ties and
Client-winning samples retain the Client distribution. If no sample selects the
Host, synchronization records a successful no-op and does not train an adapter.
Otherwise the real Qwen path:

1. loads the immutable checkpoint that produced the accepted Client package;
2. trains for exactly one public-data epoch using `0.9` supervised answer CE plus
   `0.1` selected-teacher sparse CE at temperature `1.0`;
3. requires finite losses, an optimizer step, changed LoRA tensors, an unchanged
   frozen base and fresh-load probe-logit equivalence;
4. evaluates parent and candidate on the held-out public subset;
5. runs the independent SafeFed-style LoRA-delta probe; and
6. promotes only when both gates pass and the current pointer still names the
   archived parent.

The quality rule is non-regression rather than mandatory improvement:

```text
candidate macro answer-token CE <= parent macro answer-token CE + 0.001
```

Token-weighted answer-token CE, teacher-forced exact match and teacher-forced
ROUGE-L are recorded as diagnostics. They are not averaged with the safety
score. The safety rule is independently:

```text
maliciousness probability < 0.8
```

The real safety gate requires `CLIENT_SAFEFED_PROBE_MANIFEST_PATH` to name a
local JSON manifest whose adjacent `linear_probe.safetensors` contains a
float32 linear weight and bias. The manifest fixes the exact Qwen model-profile
hash and revision, LoRA profile hash, ordered first-layer LoRA-B keys and sizes,
L2 normalization, threshold, training/validation-corpus hashes and weights
SHA-256. Pickled probe files are not accepted. The repository deliberately does
not ship unvalidated classifier weights; a missing, mismatched or tampered probe
fails real reverse training closed.

The probe follows SafeFed-LLM's first-layer LoRA-B delta and linear-probe design,
but this local candidate gate is a LegalFedLLM adaptation. It is not a claim of
reproducing SafeFed-LLM's complete federated defense or paper results. See the
[SafeFed-LLM paper](https://arxiv.org/abs/2601.07177) and
[official implementation](https://github.com/dmqx/Safe-FedLLM).

The decision record binds the job, parent and candidate hashes, both validation
records, probe artifact/report hashes, both independent gates and the observed
active pointer. A failed gate discards candidate weights while retaining the
audit. If another round has already advanced the Client, the decision is
`stale_parent`; compare-and-swap discards the stale candidate and never
overwrites the newer checkpoint. Retrying `/sync` reuses the same decision
bytes. A Host candidate rollback is not an eligibility block: the retained
accepted Host may still teach samples on which its CE is strictly lower.

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

### Knowledge-package representation measurement

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

## Real Qwen Client authoritative acceptance

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
measurements for that run and are not claimed here. Comparative model execution
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

The ordinary suite does not download or load a language model. Pinned Client,
Host, tokenizer and reverse-training acceptance cases remain explicitly opt-in.

Focused model-free Client tests can be run with:

```bash
python -m unittest -v \
  tests.test_alignment_profiles \
  tests.test_client_training \
  tests.test_client_knowledge \
  tests.test_client_real_package \
  tests.test_fedmkt_integration \
  tests.test_fedmkt_validation
```

### Focused Step 6.2 tests

The Step 6.2 suite covers the exact `0.9/0.1` answer-only objective, strict and
hash-bound probe loading, natural promotion, independent safety rejection,
forced rejection, zero-Host-teacher no-op, restart/byte idempotence and stale-
parent discard, including recovery from an interrupted candidate/result commit.
It also reruns the complete deterministic round path:

```bash
python -m unittest -v \
  tests.test_client_reverse_training \
  tests.test_reverse_integration \
  tests.test_protocol \
  tests.test_reference_dataset \
  tests.test_round
```

Two opt-in tests execute actual Qwen LoRA reverse optimization, candidate
safetensors save/reload, held-out validation, promotion, forced rejection and
restart replay. They download the pinned model if needed and require a compatible
CUDA/BF16 environment:

```bash
docker compose -p legalfedllm build client
docker compose -p legalfedllm run --rm --no-deps -T \
  -e LEGALFEDLLM_RUN_REAL_CLIENT_REVERSE_TESTS=true \
  -e LEGALFEDLLM_TOKENIZER_LOCAL_FILES_ONLY=false \
  client python -m unittest -v tests.test_client_real_reverse
```

These opt-in tests deliberately inject a deterministic benign probe so model
training and adoption can be tested before independently calibrated classifier
weights exist. They do not qualify a production probe. The model-free probe test
separately exercises the strict `probe.json` plus `linear_probe.safetensors`
contract, PEFT tensor extraction, the exact `< 0.8` boundary and tamper rejection.

For a production-style `/sync` test, place an independently trained Qwen probe
at `${CLIENT_SAFEFED_PROBE_DIR}/probe.json` with adjacent
`linear_probe.safetensors`. The manifest can be created with
`SafeFedProbeManifest.create(...)`; derive the ordered first-layer LoRA-B keys
and flattened sizes from a pinned Qwen parent adapter, bind the model/LoRA
profile hashes and pinned revision, record independent training and validation
corpus SHA-256 values, and bind the exact weights byte size and SHA-256. Do not
use a test constant or self-declared benign bias for this production-style run.

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

### Full-D^P heterogeneous alignment acceptance

The acceptance runner uses the frozen D^P, the exact pinned Qwen and Granite
tokenizer artifacts and three independently signed deterministic packages:

| Participant | Model role | Trust |
| --- | --- | ---: |
| Temporary `host` | Granite 3.3 2B validation Host | not applicable |
| `client-b` | Qwen 3 1.7B Client | 0.5 |
| `client-a` | Granite 3.3 2B Client | 1.0 |

The signed Client order is `client-b`, then `client-a`, and quorum is `2`.
Eligibility requires all hard protocol checks and `trust_score >= 0.5`. Trust is
not a teacher-selection weight. The deterministic loss schedule produces 283
Host selections, 141 `client-b` selections and 141 `client-a` selections across
565 samples; Host ties and signed-order Client ties are therefore exercised.

Run from a fully installed local environment:

```bash
python scripts/validate_fedmkt_alignment.py \
  --reference data/derived/gld2012/reference.jsonl \
  --output artifacts/fedmkt-alignment-validation.json \
  --mapping-cache artifacts/fedmkt-alignment-cache \
  --identity-dir artifacts/fedmkt-validation-identities \
  --maximum-sequence-length 4096 \
  --top-k 4
```

For the authoritative Docker environment, use the LegalFedLLM Client image and
its persistent Hugging Face cache:

```bash
mkdir -p artifacts
docker compose -p legalfedllm build client
docker compose -p legalfedllm run --rm --no-deps -T \
  -v "$PWD:/workspace" \
  -w /workspace \
  client python scripts/validate_fedmkt_alignment.py \
    --reference data/derived/gld2012/reference.jsonl \
    --output artifacts/fedmkt-alignment-validation.json \
    --mapping-cache artifacts/fedmkt-alignment-cache \
    --identity-dir artifacts/fedmkt-validation-identities \
    --hf-cache /models/huggingface \
    --maximum-sequence-length 4096 \
    --top-k 4
```

The complete JSON report and validation-only keys remain under ignored
`artifacts/`. It records exact model/tokenizer and software revisions, package
and mapping identities, artifact and cache sizes, selection and fallback counts,
per-tokenizer observed token counts, two-pass hashes, wall time and peak process
RAM. The no-truncation ceiling is 4096 because the frozen D^P includes a Granite
encoding longer than 2048 tokens. The runner still pads only to the observed
batch maximum; 4096 is a rejection bound, not an allocated tensor width.
The runner uses top-k 4, matching the accepted full-D^P Qwen package, and
retains all non-padding positions so each complete logical package remains
within the existing 25 MiB transport bound. This validation setting does not
change the protocol's configurable top-k field.
Mapping, DTW and sparse-target construction run on CPU. VRAM is recorded as not
applicable because the runner does not load model weights, perform a forward
pass or allocate CUDA tensors. Ollama and AnythingLLM are not part of this
acceptance path.

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
| `POST` | `/v1/rounds/{id}/sync` | Verify Host knowledge and complete the local reverse decision |
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
immutable reverse-training jobs and safetensors inputs
Client candidate, validation, safety and adoption decision records
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
- a strict, local, hash-bound Qwen LoRA-probe artifact boundary for real Client
  candidate adoption; and
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
CLIENT_OLLAMA_MODEL=qwen3:1.7b
HOST_SERVING_BACKEND=ollama
HOST_OLLAMA_MODEL=granite3.3:2b
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

For Bazzite/Podman with host Ollama, use:

```env
OLLAMA_BASE_URL=http://host.containers.internal:11434
```

LegalFedLLM trains through Transformers/PEFT. Automatic export of a promoted PEFT
adapter into an Ollama model profile, including the Granite adapter conversion
and import path, is future work.

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

- a recorded authoritative five-epoch, full D^P/D^V heterogeneous-model round;
- a trained and independently calibrated Qwen SafeFed-style probe artifact or a
  recorded full-D^P real reverse-Client acceptance run;
- DP-SGD or formal differential-privacy accounting;
- a learned malicious-package detector;
- HTTPS, production identity bootstrap or encrypted artifact storage;
- automatic PEFT-to-Ollama adapter publication; or
- a graphical application.

The deterministic mock backend must remain available while these real stages are
added.

## Next implementation milestones

### Authoritative real Host acceptance

Run the complete pinned Qwen-to-Granite round with the five-epoch Host schedule,
exercise both promotion and forced candidate rejection, and record wall time,
peak VRAM/RAM, communication volume, selected teachers, validation metrics and
adapter sizes.

### Authoritative real Client reverse acceptance

Train and independently validate the Qwen-specific SafeFed-style probe, then run
the complete 90/10 D^P path through natural promotion, forced rejection,
no-Host-teacher and stale-parent cases. Record wall time, peak VRAM/RAM, selected
teachers, both validation records, probe identity and adapter sizes.

### Complete heterogeneous-model round

Run multiple real Clients against a heterogeneous Host and measure wall time,
peak VRAM/RAM, communication volume, selected teaching samples, validation
behavior, adapter sizes and rollback behavior.

## Accurate project claim

LegalFedLLM demonstrates a real, pinned Qwen Client that can learn a local LoRA
adapter from private examples and turn its outputs over the complete frozen
565-sample D^P into a signed, validated and transportable Knowledge Package. It
now implements the corresponding pinned Granite Host path through exact
alignment, selective LoRA candidate training, private D^V validation, atomic
promotion or rejection, signed post-decision D^P publication and Coordinator
completion. The reverse path now constructs an immutable Client-owned job,
trains a Qwen LoRA candidate from the archived round parent, evaluates it on the
held-out public split, applies independent quality and hash-bound SafeFed-style
safety gates, and atomically promotes or discards it. Private examples,
Client-native LoRA tensors, Client validation details and per-sample D^V metrics
remain local.

The repository does **not** yet record an authoritative five-epoch full-round
acceptance, a calibrated Qwen probe/full-corpus reverse-Client acceptance,
formal differential privacy, a complete SafeFed-LLM defense or production-ready
deployment.
