# LegalFedLLM

LegalFedLLM is a mock-first implementation of the FedMKT-centered architecture
for bidirectional knowledge transfer between heterogeneous language models.
This milestone replaces the obsolete direct LoRA-delta exchange with signed
Client and Host Knowledge Packages, a separate Federated Coordinator, bounded
asynchronous rounds, filesystem persistence, candidate Host-adapter validation,
and an Ollama serving boundary.

No model download, GPU, Ollama installation, FATE-Flow deployment, or live
public server is required to test this milestone.

## Implemented milestone

```text
Client local work can happen independently
        ↓
Coordinator publishes a signed round manifest
        ↓
Clients create signed mock FedMKT Knowledge Packages independently
        ↓
Coordinator verifies, stores and safety-checks each package
        ↓
Coordinator waits for quorum or deadline
        ↓
Accepted Client set is sealed
        ↓
DualMinCE selects the best Client teacher per reference sample
        ↓
Private Host runtime creates and validates candidate ω(t+1)
        ↓
Candidate is promoted or ω(t) is retained
        ↓
Host publishes a signed Host Knowledge Package
        ↓
Clients selectively apply the Host package through a mock reverse-distillation step
```

The mock tensors, logits and losses are deterministic protocol fixtures. They
prove the service, security, persistence, selection, rollback and bidirectional
message flow; they do not represent useful model training.

## Repository layout

```text
LegalFedLLM/
├── coordinator/              Public round and package-management service
├── host/                     Private Host model/distillation service
├── client/                   Local Client agent and Ollama boundary
├── shared/
│   ├── protocol.py           Manifests, profiles and Knowledge Package schemas
│   ├── crypto.py             Ed25519 signatures and canonical SHA-256 hashing
│   ├── storage.py            Atomic cross-platform filesystem persistence
│   ├── ollama.py             Ollama HTTP connector
│   ├── fedmkt_runtime.py     Mock/real FedMKT runtime boundary
│   └── fedmkt_core/
│       ├── selection.py      Dependency-free DualMinCE selection
│       ├── safety.py         Protocol-first package safety checks
│       └── ml/               Extracted optional FATE-LLM FedMKT core
├── tests/                    Protocol, round, rollback, persistence and Ollama tests
├── scripts/demo_round.py     One complete containerized mock round
├── compose.yaml
├── requirements.txt
└── requirements-ml.txt
```

## What can be tested before real models

Yes. On Bazzite, Linux, Windows, or another system with Python, the complete
protocol-first workflow can be tested before any model integration:

- signed manifest creation and verification;
- Ed25519 Client and Host Knowledge Package signatures;
- SHA-256 payload integrity;
- nonce, duplicate and replay checks;
- package-size and timestamp checks;
- independent Client submissions;
- quorum-triggered sealing;
- deadline-based round skipping;
- deterministic DualMinCE selection;
- Host candidate promotion;
- Host rollback when validation fails;
- Host Knowledge Package publication;
- Client-side package verification and mock reverse distillation;
- persistence across Coordinator restart;
- Ollama list, inspect and generation API boundaries through mocks.

Run the tests locally:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
```

The test suite does not import PyTorch or Transformers.

## Run with Docker Compose or Podman Compose

Create the environment file and replace the three development tokens:

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
| `POST` | `/v1/rounds/{id}/knowledge` | Submit a signed Client Knowledge Package |
| `GET` | `/v1/rounds/{id}/status` | Poll round state |
| `GET` | `/v1/rounds/{id}/host-knowledge` | Download the signed Host Knowledge Package |
| `POST` | `/v1/generate` | Proxy a direct Host consultation request |

Registration requires `X-Registration-Token`. Round creation requires
`X-Admin-Token`. Public internet deployment must place HTTPS and proper
identity bootstrap in front of these APIs.

### Host runtime — private port 8002

The Host runtime owns model-specific work and adapter state:

- current Host adapter metadata;
- reference-dataset Host knowledge generation;
- candidate Host adapter creation;
- validation and rollback;
- accepted adapter versioning;
- signed Host Knowledge Package generation;
- optional Ollama-backed inference.

Its internal API is protected by `X-Internal-Token` and is not published by
`compose.yaml`.

### Client agent — port 8001

The Client agent owns local state:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Client state and backend status |
| `POST` | `/v1/register` | Register with the Coordinator |
| `POST` | `/v1/local-train` | Mock local training independent of a round manifest |
| `POST` | `/v1/participate` | Verify the manifest, snapshot local state and submit knowledge |
| `POST` | `/v1/rounds/{id}/sync` | Verify and selectively consume Host knowledge |
| `POST` | `/v1/generate` | Local mock or Ollama inference |
| `GET` | `/v1/ollama/models` | List installed Ollama models |
| `POST` | `/v1/ollama/inspect` | Inspect one Ollama model |

Raw local examples passed to `/v1/local-train` remain inside the Client service.
Only a local content hash and example count are persisted. The Coordinator
receives only the signed Knowledge Package.

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

The first Client may submit and disconnect while the Coordinator continues to
wait for the remaining Clients. The accepted set is frozen before the Host job
starts, so late arrivals cannot alter an in-progress distillation job.

## Filesystem persistence

No database is required. Each service creates its own state tree under its
configured data directory.

```text
Coordinator data
├── identity/
├── clients/
├── host/
├── rounds/
└── audit/events.jsonl

Host data
├── identity/
├── adapters/
└── rounds/

Client data
├── identity/
├── state.json
└── knowledge_cache/
```

Writes use temporary files followed by atomic replacement. The same design is
portable to Windows filesystems, Linux, Docker volumes and Podman volumes.

## Security included now

The protocol-first milestone implements:

- Ed25519 identities and signatures;
- canonical JSON signing;
- SHA-256 manifest, package, dataset and candidate-artifact hashes;
- registered Client public keys;
- signed Coordinator manifests;
- signed Client Knowledge Packages;
- signed Host Knowledge Packages;
- round-bound manifest hashes;
- nonces and timestamp skew checks;
- duplicate Client submission rejection;
- repeated package-hash rejection;
- package-size limits;
- exact model-profile verification;
- exact dataset, sample-order, top-k and alignment-profile verification;
- finite-number and shape validation;
- a minimal Knowledge Package Safety Probe;
- append-only JSONL audit events.

This does not replace HTTPS. Signatures prove package origin and integrity but do
not encrypt network traffic. A public deployment must put the Coordinator behind
a TLS reverse proxy and replace the development bootstrap tokens.

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

Install the pinned optional machine-learning environment only when moving to
real models:

```bash
python -m pip install -r requirements-ml.txt
```

The current protocol services import these dependencies lazily, so the default
mock workflow stays small.

See `shared/fedmkt_core/UPSTREAM.md`, `shared/fedmkt_core/LICENSE`, and
`THIRD_PARTY_NOTICES.md`.

## Current limitations

This milestone does not yet perform:

- real private LoRA training;
- teacher-forced inference on an actual reference dataset;
- real cross-tokenizer alignment inside a live round;
- real Host or Client knowledge distillation;
- formal differential privacy accounting;
- learned malicious-package detection;
- encrypted artifact storage;
- production authentication or certificate provisioning;
- HTTPS termination;
- a graphical Windows application;
- automatic export of accepted PEFT adapters into Ollama.

The DP fields currently enforce protocol consistency only. The safety probe is a
basic deterministic gate, not the final Safe-FedLLM-inspired detector.

## Next implementation milestone

The next stage can retain this protocol and replace only the runtime operations:

```text
mock Client local training
    → Transformers/PEFT Client LoRA training

mock knowledge generation
    → FATE-derived top-k logits and CE-loss generation

mock identity alignment
    → FATE-derived DTW/MinED token and vocabulary alignment

mock Host candidate
    → FedMKTTrainer Host LoRA distillation

mock Client sync
    → FedMKTTrainer reverse Client distillation
```

The first real-model verification should use a model pair tested by FedMKT before
adding the Windows/Ollama model-profile family.
