# LegalFedLLM

LegalFedLLM is a protocol-first implementation of a FedMKT-centered architecture
for bidirectional knowledge transfer between heterogeneous language models.

The repository now contains three connected pieces:

1. a deterministic Step-0 control plane using mock FedMKT knowledge;
2. a real generic shared-reference-dataset boundary with canonical JSONL data,
   semantic dataset identity, Coordinator-owned round snapshots, selected-Client
   download, Client verification and caching, and Host verification of both
   reference and validation data; and
3. a deterministic source-specific importer for the pinned 2012 Greek Law Digest
   thesis copy, producing canonical LegalFedLLM Q&A records for the selected
   printed-page range.

The default execution path remains mock-first. No model download, GPU, Ollama
installation, FATE-Flow deployment, or live public server is required to test the
current implementation.

The GLD importer is offline tooling. The copyrighted source PDF and generated
derivatives remain under the Git-ignored `data/` tree and are not part of the
runtime repository.

## Current development snapshot

The current round flow is:

```text
Optional canonical D^P and D^V JSONL files
        ↓
Coordinator loads and validates both datasets
        ↓
Coordinator derives the real D^P ID, hash and ordered sample IDs
        ↓
Coordinator publishes a signed round manifest
        ↓
Coordinator stores immutable round-specific D^P and D^V snapshots
        ↓
Selected Clients download D^P
        ↓
Each Client verifies the dataset ID, semantic hash and sample order
        ↓
Each Client caches the verified D^P and creates a signed mock Knowledge Package
        ↓
Coordinator verifies, stores and safety-checks each package
        ↓
Coordinator waits for quorum or deadline
        ↓
Accepted Client set is sealed
        ↓
Host receives and independently verifies D^P and D^V
        ↓
DualMinCE selects the best Client teacher per reference sample
        ↓
Private Host runtime creates and validates candidate ω(t+1)
        ↓
Candidate is promoted or ω(t) is retained
        ↓
Host publishes a signed Host Knowledge Package
        ↓
Clients selectively apply the Host package through mock reverse distillation
```

The dataset boundary is real: JSONL records, identities, hashes, snapshots,
download and cache verification operate on actual files.

The model-learning path is still mocked. The tensors, logits and losses used by
the current Knowledge Packages are deterministic protocol fixtures. They prove
the service, security, persistence, selection, rollback and bidirectional message
flow; they do not yet represent useful model training.

## Past versions

- **Early prototype:** direct LoRA-delta exchange between participants. This
  approach was removed because heterogeneous model architectures cannot safely
  aggregate arbitrary adapter parameters.
- **Step 0 protocol-first baseline:** signed manifests and Knowledge Packages,
  bounded asynchronous rounds, filesystem persistence, replay protection,
  DualMinCE selection, Host validation and rollback, Client synchronization and
  an Ollama serving boundary.
- **Step 1A reference-dataset boundary:** canonical reference samples, shared
  prompt rendering, JSONL I/O, deterministic semantic hashing, per-section
  D^P/D^V splitting, Coordinator snapshots, selected-Client delivery, Client
  verification/caching and Host D^P/D^V verification.
- **Step 1B GLD importer:** deterministic parsing of the pinned 2012 Greek Law
  Digest thesis copy, reviewed follow-up handling, subsection-context
  disambiguation, corpus auditing and freeze-ready D^P/D^V generation.

The Step 1B.2 code is implemented. Step 1 should be called fully frozen only
after the authoritative local importer run is clean and the resulting
`identity.json` values have been recorded.

## Repository layout

```text
LegalFedLLM/
├── coordinator/
│   ├── main.py                 Public federation API
│   ├── service.py              Round orchestration and Host gateway
│   └── reference_data.py       Coordinator-owned D^P/D^V boundary
├── host/
│   ├── main.py                 Private internal Host API
│   └── runtime.py              Host state, dataset cache and mock distillation
├── client/
│   ├── main.py                 Local Client API and Coordinator gateway
│   └── runtime.py              Client state, D^P cache and mock knowledge flow
├── shared/
│   ├── protocol.py             Manifests, profiles and package schemas
│   ├── reference_dataset.py    Canonical schema, JSONL I/O, hashing and splitting
│   ├── prompt.py               Shared reference-prompt renderer
│   ├── crypto.py               Ed25519 signatures and canonical SHA-256 hashing
│   ├── storage.py              Atomic cross-platform filesystem persistence
│   ├── ollama.py               Ollama HTTP connector
│   ├── fedmkt_runtime.py       Mock/real FedMKT runtime boundary
│   └── fedmkt_core/
│       ├── selection.py        Dependency-free DualMinCE selection
│       ├── safety.py           Protocol-first package safety checks
│       └── ml/                 Extracted optional FATE-LLM FedMKT core
├── tools/
│   └── datasets/
│       ├── inspect_gld_layout.py
│       │                        Offline PDF layout-inspection utility
│       └── gld_pdf_to_jsonl.py
│                                Deterministic pinned-GLD canonical importer
├── tests/
│   ├── test_gld_importer.py    GLD extraction, grouping and audit tests
│   └── ...                     Protocol, dataset, round, rollback and Ollama tests
├── scripts/demo_round.py       One complete containerized mock round
├── compose.yaml
├── requirements.txt
├── requirements-tools.txt      Optional offline dataset tooling
└── requirements-ml.txt         Optional real-model dependencies
```

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

The authoritative runtime format is UTF-8 JSONL with one sample per line.
A separate pretty JSON copy may be generated for human inspection, but JSONL
remains authoritative.

The shared prompt renderer produces:

```text
Chapter: {chapter}

Section: {section}

Question: {question}

Answer:
```

The `gold_answer` remains a separate target and is never inserted into the input
prompt.

### Canonical dataset identity

The semantic dataset hash covers:

```text
schema_version
dataset_id
dataset_version
ordered samples:
    sample_id
    chapter
    section
    question
    gold_answer
```

It does not include PDF page numbers or other source-provenance fields. Therefore,
changing the semantic sample content or sample order changes the hash, while
correcting only source-page metadata does not.

### Deterministic D^P/D^V split

Samples are grouped by `(chapter, section)` and retain their source order.

For a section containing `n` samples:

```python
if n == 1:
    D_P = all samples
    D_V = []
else:
    cut = floor(0.8 * n)
    D_P = samples[:cut]
    D_V = samples[cut:]
```

Sample IDs are not renumbered after splitting.

For GLD specifically, dependent top-level follow-up Q&A pairs are grouped by the
source-specific importer **before** this generic split runs. The original question
and answer wording is preserved and concatenated in source order. This prevents a
base question and its dependent follow-up from being separated between D^P and
D^V without changing the generic `ReferenceSample` schema.

Where a GLD section contains repeated short questions under different internal
subheadings, approved source subheadings are appended to the canonical `section`
value so that the resulting prompts remain unambiguous.

### Real-data and mock-data modes

When the Coordinator is configured with real dataset paths, it:

- loads and validates both D^P and D^V;
- requires matching dataset IDs and versions;
- rejects overlap between D^P and D^V;
- derives the manifest dataset metadata from the real D^P;
- ignores fabricated dataset metadata supplied by a round-creation request;
- snapshots both files for the specific round.

When no real dataset is configured, the existing Step-0 fixture path remains
available. In that mode, the round request supplies mock dataset metadata and the
reference-dataset download endpoint returns `204 No Content`.

## What can be tested now

The current implementation can test:

- canonical reference-sample validation;
- JSONL loading, writing and human-readable JSON generation;
- Unicode and paragraph preservation;
- duplicate and mixed-version rejection;
- canonical dataset hashing;
- deterministic per-section D^P/D^V splitting;
- deterministic GLD question-style recognition;
- multiline and black-bold GLD question extraction;
- exclusion of non-question GLD captions;
- preservation of embedded question-mark sentences inside answers;
- grouping of consecutive source questions that share one answer;
- reviewed GLD follow-up grouping without rewriting source wording;
- stable source-question IDs after grouping;
- approved subsection-context disambiguation;
- contributing-firm running-matter filtering;
- GLD corpus audits for duplicate prompts, profile contamination and scope;
- pinned GLD source SHA-256 and page-count verification;
- signed manifest creation and verification;
- Ed25519 Client and Host Knowledge Package signatures;
- SHA-256 payload integrity;
- nonce, duplicate and replay checks;
- package-size and timestamp checks;
- independent Client submissions;
- quorum-triggered sealing;
- deadline-based round skipping;
- Coordinator-owned real dataset metadata;
- immutable per-round D^P and D^V snapshots;
- selected-Client D^P download authorization;
- Client D^P hash, ID and order verification;
- rejection of tampered Client dataset downloads;
- Client round-bound dataset caching;
- Host D^P and D^V verification and caching;
- D^P/D^V dataset-version and overlap checks;
- deterministic DualMinCE selection;
- Host candidate promotion;
- Host rollback when validation fails;
- Host Knowledge Package publication;
- Client-side package verification and mock reverse distillation;
- persistence across Coordinator restart;
- authenticated Client administrative endpoints;
- immutable accepted Client package and adapter-snapshot binding;
- replay detection for authenticated packages rejected later;
- strict Host identity, round, dataset and adapter-version verification;
- explicit rejection of unintegrated real training and alignment backends;
- Ollama list, inspect and generation boundaries through mocks.

Run the lightweight protocol/runtime test environment locally:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
```

`tests/test_gld_importer.py` contains 18 importer tests and does not require opening
the real PDF. It imports PyMuPDF lazily, so the ordinary unit-test path remains
independent of the offline PDF tooling.

The previous repository-wide suite contained 45 tests; the importer adds 18 more,
so the expected total after Step 1B is 63. Confirm the final count with the command
above before recording it as a verified project result.

The default test suite does not import PyTorch or Transformers. PyMuPDF is required
only when the real PDF importer or layout-inspection utility is executed.

## Offline GLD dataset tooling

The GLD tools are separate from the LegalFedLLM runtime.

Install the optional tooling dependencies:

```bash
python -m pip install -r requirements-tools.txt
```

The tools expect the local source PDF at:

```text
data/private/greek_law_digest.pdf
```

### Layout inspection

Run:

```bash
python tools/datasets/inspect_gld_layout.py
```

The inspection utility records page text, positions, fonts, sizes, anchor matches,
the source-file SHA-256 and the PyMuPDF version under `data/derived/`. It exists to
inspect the PDF layout and extraction assumptions.

### Deterministic GLD importer

Run:

```bash
python tools/datasets/gld_pdf_to_jsonl.py
```

The importer is intentionally tied to the thesis copy of **Greek Law Digest
(2012)**:

```text
expected PDF pages: 713
expected SHA-256:
9673ee7c86b3d582e2c08e1cdd2b84f144981f31a1fe50d4216e82c5b350b77d
```

A different PDF is rejected even if it has the same title.

The initial corpus scope is fixed to complete source sections beginning at printed
page 34 and ending before `COVERED BONDS` on printed page 306. Therefore the last
included printed page is 305. The scope contains 46 listed sections: 44
Q&A-structured sections are imported and two prose-structured sections are
explicitly excluded because LegalFedLLM does not synthesize questions that are not
present in the source.

The importer:

- identifies top-level GLD questions from the reviewed source formatting;
- ignores blue captions that do not end in a question mark;
- retains valid black-bold questions;
- keeps ordinary question-mark sentences inside answers when they are not
  top-level question blocks;
- joins wrapped question lines;
- removes reviewed running headers, footers and contributing-firm profile matter;
- groups dependent follow-up Q&A pairs before D^P/D^V splitting;
- preserves original source wording and order when grouping;
- keeps source-question ordinals in sample IDs rather than renumbering after
  grouping;
- uses approved GLD subsection headings to disambiguate otherwise identical
  prompts;
- records reviewed follow-up overrides;
- audits the final canonical corpus for duplicate prompts, remaining firm/profile
  text and out-of-scope samples;
- preserves source answers that consist only of an internal cross-reference and
  records them as warnings rather than inventing replacement text.

If unresolved extraction issues or follow-up candidates remain, the run stops and
writes:

```text
data/derived/gld2012/
├── candidate_all.jsonl
├── candidate_all.json
└── review.json
```

It deliberately removes stale final corpus files in that state.

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

`all.jsonl`, `reference.jsonl` and `validation.jsonl` are the authoritative
machine-readable datasets. The corresponding `.json` files are generated
human-readable copies.

`review.json` records source identity, extraction-tool version, scope, per-section
boundaries, warnings, reviewed follow-up decisions and the corpus audit.
`identity.json` records the source identity plus semantic identities/hashes for the
full corpus, D^P and D^V.

The `data/` directory is ignored by Git. The GLD source PDF and generated
derivatives remain local and must not be committed unless the required
distribution permission is obtained.

PyMuPDF is an optional offline dependency and is dual-licensed under the GNU
AGPL-3.0 or an Artifex commercial licence. See `THIRD_PARTY_NOTICES.md`.

## Run with Docker Compose or Podman Compose

Create the environment file and replace the development tokens:

```bash
cp .env.example .env
```

Start the services:

```bash
docker compose up --build -d
```

On Bazzite, the corresponding command is normally:

```bash
podman compose up --build -d
```

Check the public services:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8001/health
```

Run one complete mock round:

```bash
. .env
python scripts/demo_round.py
```

Reset all prototype state:

```bash
docker compose down -v
```

or:

```bash
podman compose down -v
```

Only the Coordinator and Client agent publish host ports. The Host runtime is
reachable only through the private Compose network.

## Configure real reference and validation JSONL files

Set both paths for the Coordinator:

```env
COORDINATOR_REFERENCE_DATASET_PATH=data/derived/gld2012/reference.jsonl
COORDINATOR_VALIDATION_DATASET_PATH=data/derived/gld2012/validation.jsonl
```

Both variables must be configured together. Startup fails if only one is set or
if the files are malformed, inconsistent or overlapping.

The signed manifest binds the round to D^P through:

```text
reference_dataset_id
reference_dataset_hash
ordered sample_ids
```

D^V is retained by the Coordinator and Host and is not exposed through the
public Client download endpoint.

## Service boundaries

### Federated Coordinator — port 8000

The Coordinator owns the public federation protocol:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Coordinator status |
| `GET` | `/v1/identity` | Coordinator and Host public keys |
| `POST` | `/v1/clients/register` | Register a Client model profile and public key |
| `POST` | `/v1/rounds` | Create and sign a round manifest |
| `GET` | `/v1/rounds/current` | Retrieve the active manifest |
| `GET` | `/v1/rounds/{id}/manifest` | Retrieve one manifest |
| `GET` | `/v1/rounds/{id}/reference-dataset` | Download the round D^P as JSONL |
| `POST` | `/v1/rounds/{id}/knowledge` | Submit a signed Client Knowledge Package |
| `GET` | `/v1/rounds/{id}/status` | Poll round state |
| `GET` | `/v1/rounds/{id}/host-knowledge` | Download the signed Host Knowledge Package |
| `POST` | `/v1/generate` | Proxy a direct Host consultation request |

The D^P endpoint requires a valid registration token and a registered Client ID.
The Client must also be selected in the signed round manifest. D^V has no public
download endpoint.

Registration requires `X-Registration-Token`. Round creation requires
`X-Admin-Token`. Public internet deployment must place HTTPS and stronger identity
bootstrap in front of these APIs.

### Host runtime — private port 8002

The Host runtime owns:

- current Host adapter metadata;
- private D^P and D^V loading;
- independent dataset verification;
- per-round dataset caching;
- reference-dataset Host knowledge generation;
- candidate Host adapter creation;
- validation and rollback;
- accepted adapter versioning;
- signed Host Knowledge Package generation;
- optional Ollama-backed inference.

Its private API includes:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/internal/v1/identity` | Host identity and model profile |
| `POST` | `/internal/v1/reference-data` | Load and verify round D^P and D^V |
| `POST` | `/internal/v1/reference-knowledge` | Create Host reference knowledge |
| `POST` | `/internal/v1/distill` | Run the current mock distillation boundary |

These endpoints are protected by `X-Internal-Token` and are not published by
`compose.yaml`.

### Client agent — port 8001

The Client agent owns local state:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Client state and backend status |
| `POST` | `/v1/register` | Register with the Coordinator |
| `POST` | `/v1/local-train` | Mock local training independent of a round |
| `POST` | `/v1/participate` | Verify manifest, download/cache D^P and submit knowledge |
| `POST` | `/v1/rounds/{id}/sync` | Verify and selectively consume Host knowledge |
| `POST` | `/v1/generate` | Local mock or Ollama inference |
| `GET` | `/v1/ollama/models` | List installed Ollama models |
| `POST` | `/v1/ollama/inspect` | Inspect one Ollama model |

The administrative endpoints require `X-Client-Admin-Token`. `/health` and
`/v1/generate` remain outside that administrative gate.

Raw examples passed to `/v1/local-train` remain inside the Client service. Only a
local content hash and example count are persisted. The Coordinator receives only
the signed Knowledge Package.

## Bounded asynchronous round state

```text
COLLECTING
    ↓ trusted quorum reached
SEALED
    ↓
DISTILLING
    ↓
COMPLETED
```

A round becomes `SKIPPED` when its deadline passes without the required trusted
quorum. It becomes `ABORTED` if a sealed Host job fails. Late packages and second
submissions from the same Client are rejected.

The first Client may submit and disconnect while the Coordinator waits for the
remaining Clients. The accepted set is frozen before the Host job starts, so late
arrivals cannot alter an in-progress distillation job.

## Filesystem persistence

No database is required. Each service creates its own state tree under its
configured data directory.

```text
Coordinator data
├── identity/
├── clients/
├── host/
├── rounds/
│   └── <round-id>/
│       ├── manifest.json
│       ├── state.json
│       ├── datasets/
│       │   ├── reference.jsonl
│       │   ├── validation.jsonl
│       │   └── identity.json
│       ├── submissions/
│       ├── safety/
│       └── ...
└── audit/events.jsonl

Host data
├── identity/
├── adapters/
└── rounds/
    └── <round-id>/
        ├── datasets/
        │   ├── reference.jsonl
        │   ├── validation.jsonl
        │   └── identity.json
        └── ...

Client data
├── identity/
├── state.json
├── reference_datasets/
│   └── <round-id>/
│       ├── reference.jsonl
│       └── identity.json
├── knowledge_cache/
│   ├── pending/
│   ├── accepted/
│   └── receipts/
└── adapter_snapshots/
    ├── pending/
    └── accepted/
```

A downloaded Client dataset is written to a temporary path and accepted only
after schema, dataset ID, semantic hash and sample-order verification. The cache
identity is bound to the round ID and manifest hash.

A Client Knowledge Package is first written as pending state. Only after a
matching Coordinator receipt is returned is the exact package and its round-bound
adapter snapshot committed as accepted state. Accepted package/snapshot pairs are
immutable for that round.

Writes use temporary files followed by atomic replacement. Direct execution
defaults to `./data/...`; Compose overrides these paths with `/data/...` volume
mounts.

## Security included now

The current implementation includes:

- Ed25519 identities and signatures;
- canonical JSON signing;
- SHA-256 manifest, package, dataset and candidate-artifact hashes;
- registered Client public keys;
- signed Coordinator manifests;
- signed Client and Host Knowledge Packages;
- round-bound manifest hashes;
- nonces and timestamp-skew checks;
- duplicate Client submission rejection;
- replay tracking for seen nonces and package hashes;
- package-size limits;
- exact model-profile verification;
- exact dataset ID, semantic hash and sample-order verification;
- selected-Client authorization for D^P download;
- Client cache binding to the round and manifest;
- Host verification of D^P against the signed manifest;
- Host verification of D^V identity, version and separation from D^P;
- strict Host identity, signature, manifest and adapter-version verification;
- authenticated Client administrative endpoints;
- immutable accepted Client package and adapter-snapshot binding;
- finite-number and shape validation;
- a minimal Knowledge Package Safety Probe;
- append-only JSONL audit events.

This does not replace HTTPS. Signatures prove package origin and integrity but do
not encrypt network traffic. The current registration and internal tokens are
development credentials, not production-grade per-service identity.

## Ollama boundary and Windows baseline

The default backend remains `mock`. To use an existing Ollama installation for
normal inference, set:

```env
CLIENT_SERVING_BACKEND=ollama
CLIENT_OLLAMA_MODEL=<installed model name>
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

The intended first Windows environment is:

```text
Windows 10/11
├── native Ollama
└── Docker Desktop with WSL2 Linux containers
    ├── Coordinator
    ├── Host runtime
    └── Client agent
```

For Bazzite or another Podman host, use:

```env
OLLAMA_BASE_URL=http://host.containers.internal:11434
```

Ollama is currently a serving boundary only. The real FedMKT training path will
use Transformers and PEFT, then publish accepted model-specific adapters into a
verified Ollama model profile. An arbitrary Ollama model is not automatically a
supported training model.

## Extracted FedMKT core

The optional machine-learning files under `shared/fedmkt_core/ml/` are adapted
from FATE-LLM commit:

```text
0c63377e468f0f62a9bdf5fb32424688b9478553
```

Included components are:

- `FedMKTTrainer`;
- `DataCollatorForFedMKT`;
- top-k logit and CE-loss generation;
- token alignment;
- vocabulary mapping;
- FedMKT constants.

FATE Context, FATE communication roles, FATE-Flow and FATE aggregation wrappers
are not included. LegalFedLLM supplies the HTTP, security, storage and round
layers.

Install the optional machine-learning environment only when moving to real models:

```bash
python -m pip install -r requirements-ml.txt
```

The current protocol services import these dependencies lazily, so the default
mock workflow stays small.

See `shared/fedmkt_core/UPSTREAM.md`, `shared/fedmkt_core/LICENSE`, and
`THIRD_PARTY_NOTICES.md`.

## Current limitations

The repository does not yet perform:

- real private LoRA training;
- teacher-forced model inference on D^P;
- real top-k logit and CE-loss extraction;
- real cross-tokenizer alignment inside a live round;
- real Host or Client knowledge distillation;
- scalable binary Knowledge Package artifact transfer;
- formal differential-privacy accounting;
- learned malicious-package detection;
- encrypted artifact storage;
- production authentication or certificate provisioning;
- HTTPS termination;
- a graphical Windows application;
- automatic export of accepted PEFT adapters into Ollama.

The GLD-specific deterministic importer is implemented, but generated GLD content
is intentionally not committed to the public repository. Until the authoritative
local Step 1B.2 run is completed and its clean `review.json` and `identity.json`
values are recorded, the project should not claim a final frozen thesis corpus
identity.

The current D^P/D^V transfer and verification boundary is real, but the Host and
Client still generate deterministic mock knowledge rather than model-derived
knowledge.

The DP fields currently enforce protocol consistency only. The safety probe is a
basic deterministic gate, not the final Safe-FedLLM-inspired detector.

## Next implementation milestones

### Finish the Step 1 corpus freeze

The generic dataset boundary and the GLD importer are implemented. The remaining
Step 1 freeze procedure is operational rather than architectural:

```text
run the pinned GLD importer locally
        ↓
require review.json status == clean
        ↓
require zero unresolved follow-up candidates
        ↓
require corpus_audit status == clean
        ↓
inspect representative records and section boundaries
        ↓
record all/reference/validation identities from identity.json
        ↓
record the verified repository-wide test result
        ↓
Step 1 complete
```

Generated GLD source data remains local and Git-ignored.

### Step 2 — Scalable Knowledge Package artifacts

The current protocol keeps deterministic numerical arrays directly in signed
JSON. Before real-model scale, large tensors should move behind a signed metadata
envelope and a constrained binary artifact such as `safetensors` or strictly
validated NumPy `.npz`. Arbitrary pickle/PyTorch object deserialization from
Clients must not be accepted.

### Step 3 — First real Client model

Connect one small Client runtime using pinned Transformers/tokenizer revisions and
PEFT LoRA:

```text
private Client examples
    → PEFT LoRA training
    → real D^P inference
    → top-k token/logit extraction
    → real cross-entropy losses
    → signed real Client Knowledge Package
```

### Step 4 — FedMKT parity and token alignment

Connect and verify the extracted FedMKT components:

```text
Client-to-Host alignment
Host-to-Client alignment
vocabulary mapping
DataCollatorForFedMKT
FedMKTTrainer behavior
DualMinCE parity
```

### Step 5 — Real Host distillation

Replace the mock Host candidate path with real baseline inference, selective
FedMKT LoRA distillation, D^V validation and real adapter promotion or rollback.

### Step 6 — Real reverse Client distillation

Verify the real Host Knowledge Package, align Host outputs into the Client token
space, select samples where the Host CE is lower and distil those targets into the
Client-specific LoRA.

### Step 7 — Complete heterogeneous-model round

Run multiple real Clients against a heterogeneous Host and measure execution
time, VRAM/RAM, communication volume, selected teaching samples, validation
behavior, adapter sizes and rollback behavior.

The deterministic mock backend should remain available throughout these stages.

## Current project claim

The repository currently demonstrates:

- a working protocol-first control plane for FedMKT-style heterogeneous knowledge
  transfer;
- a canonical and cryptographically identified shared-reference-dataset format;
- a deterministic source-specific importer for the pinned 2012 Greek Law Digest
  thesis copy;
- reviewed handling of GLD question styles, dependent follow-ups and internal
  subsection context;
- corpus-level auditing before GLD D^P/D^V files are accepted;
- Coordinator-owned D^P/D^V round snapshots;
- authorized D^P delivery to selected Clients;
- independent Client and Host dataset verification and caching;
- deterministic mock bidirectional FedMKT message flow.

It does **not** yet demonstrate a completed real-model FedMKT round, formal
differential privacy, a complete Safe-FedLLM defense, or production-ready
deployment.

A final frozen GLD corpus should be claimed only after the authoritative local
Step 1B.2 run is clean and its semantic identities/hashes are recorded.
