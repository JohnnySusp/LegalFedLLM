# LegalFedLLM

LegalFedLLM is a protocol-first proof of concept for **bidirectional federated
knowledge transfer between heterogeneous language models** in legal-domain
workloads.

The project does not aggregate LoRA tensors across incompatible models. Each
participant keeps its own base model and model-native LoRA adapter local. What
crosses the federation boundary is a signed **Knowledge Package** produced over
a common reference dataset: retained token IDs and logits, answer-token loss
evidence, model/tokenizer identities, dataset identities, hashes, signatures,
and related provenance.

The real split-machine topology currently exercised by the repository is:

```text
local machine                                      remote NVIDIA A40 container

private Client JSONL
      ↓
Qwen3 1.7B + PEFT LoRA
      ↓
D^P inference
      ↓
signed Client Knowledge Package
      ───────────────────────────────────────────→ Coordinator
                                                    ↓
                                            verification + safety
                                                    ↓
                                         Qwen ↔ Nemo DTW alignment
                                                    ↓
                                           DualMinCE selection
                                                    ↓
                                         Mistral Nemo Host LoRA
                                                    ↓
                                         hidden D^V validation
                                                    ↓
                                        promote or roll back Host
                                                    ↓
                                      signed Host Knowledge Package
      ←─────────────────────────────────────────────┘
      ↓
automatic Client sync
      ↓
Host → Qwen reverse alignment / selective distillation
      ↓
Qwen candidate quality + safety gates
      ↓
promote, reject, or discard stale candidate
```

No Client LoRA tensor is sent to the Coordinator or Host. Raw private Client
training examples remain on the Client machine.

> **Research boundary:** LegalFedLLM is a thesis proof of concept, not a
> production federated-learning platform. The repository contains real-model
> training and heterogeneous knowledge-transfer paths, but it does not claim
> formal differential privacy, production-calibrated poisoning detection,
> production identity management, or encrypted transport/storage.

## Status at a glance

| Capability | Current status |
| --- | --- |
| Deterministic protocol/mock path | Implemented and regression-tested |
| Canonical shared-reference dataset boundary | Implemented |
| Signed Knowledge Package transport | Implemented |
| Qwen3 1.7B real Client LoRA training | Implemented and real-tested |
| Qwen real D^P Knowledge Package generation | Implemented and real-tested |
| Qwen → Mistral Nemo heterogeneous DTW alignment | Implemented and real-tested |
| SafeFed-inspired Knowledge Package screening/trust | Implemented as a PoC defense-in-depth layer |
| Mistral Nemo real Host LoRA training | Implemented and real-tested on NVIDIA A40 |
| Hidden D^V Host validation and promotion/rollback | Implemented and real-tested |
| Automatic Host → Qwen reverse distillation | Implemented and real-tested |
| Exact submission acknowledgement reconciliation | Implemented and regression-tested |
| Unattended split-machine tmux orchestration/evidence | Implemented and real-tested |
| Real multi-Client round | Not yet demonstrated |
| Mixed Qwen + Granite Clients in one live manifest | Not yet supported |
| Automatic promoted-PEFT → Ollama publication | Not implemented |
| Production-calibrated malicious-package/LoRA classifier | Not complete |
| Formal DP-SGD/privacy guarantee | Not claimed |

The normal Coordinator quorum policy is **majority with a minimum trusted quorum
of 2**. A one-Client quorum override exists only for controlled proof-of-concept
testing.

## Verified real cross-machine round

A fresh unattended bidirectional run has been verified with:

- a local **Qwen/Qwen3-1.7B** Client;
- a remote **mistralai/Mistral-Nemo-Instruct-2407** Host;
- the Coordinator colocated with the Host on an NVIDIA A40 container;
- one private-data Client epoch;
- five Host reference-data epochs;
- one reverse Client reference-data epoch;
- DTW alignment;
- DualMinCE teacher selection;
- top-k `4`;
- maximum sequence length `4096` with truncation rejected; and
- `chat_sft_answer_only_v1` supervision.

The verified run produced:

| Measurement | Result |
| --- | --- |
| Coordinator terminal state | `COMPLETED` |
| Client Knowledge Package submission | normal `201 Created` acknowledgement |
| Manual acknowledgement recovery | not required |
| Client D^P samples | 565 |
| Client stored token rows | 133,336 |
| Client Knowledge Artifact | 6,943,336 bytes |
| Post-alignment Client safety | accepted, trust score `0.33` |
| Forward Host-teacher selections | 565 / 565 |
| Forward Qwen-teacher selections | 0 / 565 |
| Host reference-data epochs | 5 |
| Host optimizer steps | 710 |
| Host adapter | 0 → 1, promoted |
| D^V macro mean answer-token CE | `2.4609236204 → 1.9685784675` |
| Required D^V improvement | `0.001` |
| Observed D^V improvement | `0.4923451529` |
| Reverse Host-teacher samples | 508 |
| Qwen training adapter | 1 → 2 |
| Reverse candidate decision | `candidate_accepted` |
| Client `last_completed_round` | `round-000001` |
| tmux evidence result | `PASS` |

The Qwen package was not rejected by the safety layer. It was accepted, aligned,
and eligible for selective distillation. However, the Host baseline had the
lower answer-token CE on every D^P sample, so DualMinCE selected the Host as the
forward teacher on all 565 samples.

That means the run verifies the full forward protocol, safety, tokenizer
alignment, selection, Host training, D^V validation, publication, automatic
sync, and reverse path. It does **not** show that Qwen knowledge caused the Host
validation improvement, because the Qwen Client was selected as teacher on
`0 / 565` forward samples.

The reverse direction is different: the promoted Host was selected as teacher
on 508 transfer samples, so the run contains substantive selected-teacher
**Host-to-Qwen** transfer.

The numerical values above describe one experiment. They are not protocol
constants or general model-performance claims.

## Architecture and protocol boundary

LegalFedLLM follows the FedMKT/FATE-LLM idea of transferring model behavior
instead of averaging heterogeneous model parameters.

A normal round is:

```text
1. Coordinator signs a round manifest.
2. Selected Clients verify the manifest and exact D^P identity/order.
3. Each Client trains its own local LoRA on private examples.
4. Each Client runs teacher-forced inference over D^P.
5. Each Client signs and uploads a Knowledge Package + safetensors artifact.
6. Coordinator verifies transport, identity, replay, dataset and safety rules.
7. Eligible Client outputs are aligned into the Host tokenizer space.
8. DualMinCE chooses one teacher per D^P sample.
9. Host trains a model-native LoRA candidate from sparse targets.
10. Host evaluates active and candidate adapters on hidden D^V.
11. Host promotes or rolls back the candidate.
12. Host publishes a signed post-decision Knowledge Package over D^P.
13. Clients verify the Host package and execute Client-owned reverse alignment.
14. A Client trains a local reverse candidate only when Host-teacher samples exist.
15. Independent Client quality and safety gates decide local adoption.
16. Client commits the round only after the reverse decision completes.
```

The core architectural rule is:

```text
model-specific weights stay model-specific
knowledge crosses the federation boundary
```

There is no heterogeneous LoRA averaging step.

## Implemented components

The repository currently includes:

- signed round manifests and Knowledge Packages;
- Ed25519 service/Client identities;
- filesystem-backed round and checkpoint state;
- deterministic replay/nonce protection;
- canonical reference-dataset hashing and ordered sample identities;
- deterministic D^P/D^V splitting;
- a reviewed Greek Law Digest importer for the thesis dataset;
- real Transformers/PEFT Client and Host training backends;
- answer-only causal supervision;
- `safetensors` Knowledge Artifacts;
- bounded multipart package transport;
- exact tokenizer-artifact validation;
- demand-driven vocabulary mapping;
- FedMKT-compatible DTW token alignment;
- sparse top-k trainer targets instead of full-vocabulary dense targets;
- minimum-CE / DualMinCE teacher selection;
- SafeFed-inspired package plausibility and post-alignment trust analysis;
- trust-gated selective distillation;
- Host D^V validation and atomic promotion/rollback;
- immutable Client reverse-training jobs;
- Qwen reverse LoRA candidate training and adoption gates;
- exact accepted-submission receipt reconciliation;
- local Client Ollama serving integration; and
- split-machine tmux orchestration with preserved run evidence.

The deterministic mock path remains available for protocol and failure-policy
regression tests. Mock loss/safety values are fixtures; they are not real-model
measurements.

## Pinned model profiles

### Clients

| Profile | Model | Revision | Current role/status |
| --- | --- | --- | --- |
| `qwen3-1.7b-lora-v1` | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Real Client path with private training, D^P package generation and reverse training verified |
| `granite-3.3-2b-instruct-client-lora-v1` | `ibm-granite/granite-3.3-2b-instruct` | `652c333dc5066f2a1764854a1bcd0ce67163d74f` | Pinned Client identity/alignment profile; no authoritative full private-training round yet |

The Qwen profile uses `Qwen3ForCausalLM`, `Qwen2TokenizerFast`, vocabulary size
151,936 and the non-thinking Qwen chat-template mode. The Granite profile uses
`GraniteForCausalLM`, `GPT2TokenizerFast` and vocabulary size 49,159.

Both Client profiles use the current PoC LoRA contract:

```text
rank:             8
alpha:            16
dropout:          0.05
target modules:   q_proj, k_proj, v_proj, o_proj
bias:             none
task:             CAUSAL_LM
```

### Hosts

| Profile | Model | Revision | Current role/status |
| --- | --- | --- | --- |
| `mistral-nemo-instruct-2407-host-lora-v1` | `mistralai/Mistral-Nemo-Instruct-2407` | `04d8a90549d23fc6bd7f642064003592df51e9b3` | Current remote Host with real A40 training/validation path verified |
| `granite-3.3-2b-instruct-host-lora-v1` | `ibm-granite/granite-3.3-2b-instruct` | `652c333dc5066f2a1764854a1bcd0ce67163d74f` | Retained compatibility Host profile |

The pinned Mistral Nemo Host uses `MistralForCausalLM`,
`PreTrainedTokenizerFast`, vocabulary size 131,072 and the same rank-8 LoRA
target modules. Its serving backend is deliberately `mock`; its training backend
is real Transformers/PEFT.

`compose.host-ml.yaml` remains the local Granite ML Compose profile. The Mistral
Nemo Host path runs directly from a Python virtual environment on the remote A40
container.

## Pinned alignment profiles

LegalFedLLM recognizes four exact bidirectional tokenizer-alignment contracts:

| Client | Host | Alignment ID | Status |
| --- | --- | --- | --- |
| Qwen3 1.7B | Granite 3.3 2B | `dtw:qwen3-1.7b--granite3.3-2b-v1` | Supported compatibility/validation profile |
| Granite 3.3 2B | Granite 3.3 2B | `dtw:granite3.3-2b-client--granite3.3-2b-host-v1` | Supported compatibility/validation profile |
| Qwen3 1.7B | Mistral Nemo | `dtw:qwen3-1.7b--mistral-nemo-instruct-2407-v1` | Real heterogeneous profile verified |
| Granite 3.3 2B | Mistral Nemo | `dtw:granite3.3-2b-client--mistral-nemo-instruct-2407-v1` | Supported profile; no full real round yet |

These contracts pin model/tokenizer revisions, tokenizer artifact hashes,
special-token state, vocabulary ranges, padding behavior, chat-template hashes,
and word-boundary rules. Unknown or mismatched profiles fail closed.

A live manifest currently carries one alignment identity. A Qwen Client and a
Granite Client therefore cannot yet participate together in the same live round;
per-Client alignment identities in one heterogeneous manifest remain future work.

## Repository layout

```text
LegalFedLLM/
├── client/                     Client API, training, package generation,
│                               reverse training and local safety probe
├── coordinator/                round lifecycle, quorum, package intake,
│                               safety/trust, Host integration and persistence
├── host/                       Host API, pinned profiles, real PEFT training,
│                               validation and post-decision publication
├── shared/                     protocol, crypto, datasets, artifacts,
│   └── fedmkt_core/            FedMKT parity/adaptation, alignment, selection,
│       └── ml/                 optional upstream-derived ML components
├── config/
│   ├── container.env.example   remote Host/Coordinator template
│   └── clients.env.example     split Client template
├── scripts/
│   ├── bootstrap.py            role-aware private environment bootstrap
│   ├── create_remote_round.py  Host-side split-round creation
│   ├── run_host_stack.py       direct Host + Coordinator process launcher
│   ├── run_remote_round.py     Client register/train/participate/sync driver
│   ├── run_split_round_tmux.sh split-machine orchestration/evidence dashboard
│   ├── validate_fedmkt_alignment.py
│   ├── measure_knowledge_packages.py
│   └── demo_round.py
├── tests/                      model-free and opt-in real-model regression tests
├── tools/datasets/             source-specific GLD inspection/import tooling
├── compose.yaml                local development stack
├── compose.clients.yaml        role-separated Qwen/Granite Client stack
├── compose.host-ml.yaml        local Granite Host ML override
├── requirements.txt
└── THIRD_PARTY_NOTICES.md
```

Generated/runtime state is intentionally outside the tracked source boundary.
`.gitignore` excludes `.env*`, `data/`, `artifacts/`, logs, PEM/private-key files,
virtual environments, and local delivery archives.

## Clean-clone and private-state contract

A clean checkout contains source code, model/profile contracts, Compose files,
environment templates, tests, and bootstrap tooling. It does not contain:

- deployment secrets;
- Ed25519 private keys;
- private Client training examples;
- downloaded model weights;
- trained adapters;
- runtime round state;
- the copyrighted Greek Law Digest PDF; or
- generated thesis D^P/D^V datasets.

Create role-specific private environment files with `scripts/bootstrap.py`.
The command creates the file only when absent and reuses it rather than silently
overwriting it.

Host/Coordinator example:

```bash
python scripts/bootstrap.py host \
  --output .env.host \
  --runtime-root /scratch/legalfedllm-test
```

Client example, using the Host-issued registration token:

```bash
python scripts/bootstrap.py client \
  --output .env.remote-client \
  --registration-token '<host-issued registration token>'
```

For controlled one-Client proof-of-concept testing only, the Host bootstrap
supports:

```bash
python scripts/bootstrap.py host \
  --output .env.host \
  --runtime-root /scratch/legalfedllm-test \
  --trusted-quorum-override 1
```

Do not use the one-Client override as the normal federation configuration.

## Private Client training data

Real Client training reads local UTF-8 JSONL. Each record has the logical form:

```json
{
  "schema_version": "1.0",
  "example_id": "private-example-001",
  "prompt": "A private local instruction or question.",
  "answer": "The private local target answer."
}
```

The Client enforces unique/non-empty IDs, non-empty prompt/answer strings,
deterministic order and hashing, a supported schema, and answer-only
supervision. Overlength examples fail rather than being silently truncated.

For the role-separated Client Compose stack, the configured private directory is
mounted read-only and the training file is expected as `/private/train.jsonl`.
The raw private examples are never written to Coordinator storage or included in
a Knowledge Package.

## Reference dataset boundary

A canonical reference record contains:

```json
{
  "schema_version": 1,
  "dataset_id": "example-reference",
  "dataset_version": "v1",
  "sample_id": "example-ch001-s001-q001",
  "chapter": "Example Chapter",
  "section": "Example Section",
  "question": "What is the question?",
  "gold_answer": "The reference target answer.",
  "source": {
    "document_id": "example-document",
    "page_start": 10,
    "page_end": 11
  }
}
```

The authoritative runtime representation is UTF-8 JSONL with one sample per
line. The semantic hash binds the schema, dataset identity, and ordered
`sample_id`, `chapter`, `section`, `question`, and `gold_answer` values. Source
page metadata is provenance and is not part of the semantic hash.

The generic split groups samples by `(chapter, section)` and preserves source
order. A one-sample section belongs entirely to D^P. Otherwise D^P receives the
first `floor(0.8 * n)` samples and D^V receives the remainder. Source-specific
GLD follow-up grouping is resolved before the generic split.

### Thesis GLD identities

The reviewed thesis dataset is derived from a pinned 713-page 2012 Greek Law
Digest source copy.

| Dataset | Samples | Semantic SHA-256 |
| --- | ---: | --- |
| Complete canonical corpus | 738 | `cf5c81dcecaab58848c1afb0e99f86bcf5fd32823c2aaee34a65f8a3dc21d49` |
| D^P shared reference set | 565 | `5d855a429d43b70eb146aeb11cda1f675c05d6465bea0792796fdcd8d6ceb231` |
| D^V hidden validation set | 173 | `1e40a74799b9900ff8b9a9e05dd379fd0c00226370625f7da1fdca13142b83b5` |

D^P is distributed only to selected Clients for a signed round. D^V has no
public Client download endpoint and remains on the Coordinator/Host side for
candidate validation.

The Greek Law Digest source is private/copyrighted. Neither the source PDF nor
the generated thesis dataset belongs in the public repository.

### Offline GLD tooling

Source-specific inspection/import code remains outside the generic dataset
boundary under `tools/datasets/`. The workflow is deliberately review-oriented:
ambiguous structure should produce deterministic warnings or blocking review
rather than silent guessing.

Generated files should be regenerated from importer rules rather than manually
edited. Provenance, source identity, sample count, and semantic hashes must stay
reproducible.

## Prompt and answer-only supervision

Client and Host training use the shared supervision label:

```text
chat_sft_answer_only_v1
```

Prompt rendering is model/tokenizer-specific, but loss is computed only over the
answer portion. The same answer-token boundary is used when producing CE
evidence for teacher selection and validation.

Overlength training/reference rows are rejected when the active profile uses the
`reject` truncation policy.

## Knowledge Package and Artifact contract

A Knowledge Package is a signed JSON envelope plus a bounded `safetensors`
artifact. The manifest binds at least:

- round identity;
- sender identity and public-key trust context;
- model and tokenizer profile;
- adapter identity;
- exact alignment profile;
- reference-dataset identity and ordered sample set;
- artifact byte size and SHA-256;
- top-k setting;
- nonce/replay state; and
- signature.

The artifact stores sparse knowledge rather than full dense vocabulary logits.
For each answer-token row it retains top-k token IDs/logits plus evidence needed
for exact CE checks and alignment.

Coordinator validation checks tensor names, dtypes, shapes, offsets,
finite-valued data, vocabulary bounds, top-k consistency, sample offsets,
artifact size/hash, and package/manifest identities before the package can be
accepted.

## FedMKT alignment and teacher selection

Different tokenizers cannot compare token IDs directly. LegalFedLLM therefore
maps tokenizer pieces into a shared word/character representation and applies a
DTW-based alignment adapted from FedMKT/FATE-LLM.

The implementation keeps the alignment deterministic and cacheable:

```text
Client sparse logits
      ↓
Client token strings / word-boundary mapping
      ↓
DTW alignment against Host tokenization
      ↓
aligned sparse teacher evidence
      ↓
Host-vs-teacher answer-token CE comparison
      ↓
DualMinCE teacher choice per sample
```

The Client does not automatically become a teacher merely because its package
was accepted. Safety eligibility and selective CE comparison are separate
conditions. If the Host is better on a sample, the Host remains the teacher for
that sample.

## Safety model

LegalFedLLM adapts the defense-in-depth idea of Safe-FedLLM to behavioral
Knowledge Packages rather than treating heterogeneous Client parameters as
aggregatable updates.

The safety path includes:

```text
structural validation
      ↓
pre-alignment plausibility checks
      ↓
post-alignment disagreement analysis
      ↓
trust score / eligibility
      ↓
trust-gated teacher selection
      ↓
hidden D^V Host outcome validation
      ↓
promotion or rollback
```

Pre-alignment checks are intentionally cheap and deterministic. They catch
malformed/non-finite artifacts, invalid token IDs/order, and gross pathological
concentration/repetition without assuming that natural Qwen/Nemo vocabulary
heterogeneity is malicious.

Repeated-frequency calculations use linear-time counting rather than repeated
full-list scans. Artifact loading and CPU-heavy pre-alignment inspection run off
the Coordinator event-loop thread so health/status/receipt endpoints remain
responsive while safety inspection is in progress. Safety still gates durable
acceptance; the Coordinator does not report an accepted receipt before the
package has passed the required checks and been persisted.

Post-alignment checks compare behavior after heterogeneous token spaces have
been normalized enough for meaningful disagreement analysis. Trust can gate or
down-weight selective distillation.

Hidden D^V validation is an additional outcome barrier, not a complete poisoning
proof. Targeted/backdoor behavior outside D^V coverage can still evade aggregate
validation metrics, so the safety layers are complementary rather than
interchangeable.

The Client reverse path also contains local candidate quality/safety gates. Any
LoRA-delta probe should be described as an experimental LegalFedLLM heuristic,
not as literal SafeFed-LMM equivalence.

## Host training and D^V promotion

After package acceptance and alignment, the Coordinator constructs sparse Host
trainer inputs. The Host trains a model-native LoRA candidate; no Client LoRA
weights are inserted into the Host.

The Host then evaluates the active adapter and candidate on hidden D^V using the
same answer-only loss contract. The decision is fail-closed:

```text
candidate improves by required margin → promote atomically
candidate does not improve            → keep active adapter
validation/training failure            → do not promote
```

The post-decision Host Knowledge Package is generated from the active adapter
after that decision, so Clients sync from the actual promoted/retained Host
state.

## Client reverse distillation and adoption

After Host publication, the Client:

1. downloads and verifies the signed Host package;
2. checks exact round/reference/alignment identities;
3. aligns Host sparse knowledge into the Client tokenizer space;
4. selects Host-teacher samples using the reverse CE rule;
5. creates an immutable reverse-training job;
6. trains a model-native Client LoRA candidate when transfer samples exist;
7. evaluates local quality/safety gates; and
8. commits or rejects the candidate before marking the round complete.

Reverse training is Client-owned. The Coordinator/Host never installs a Client
adapter.

## Submission acknowledgement and retry semantics

Knowledge submission is transactional from the Client's point of view. The
Client keeps the exact pending package, artifact, and adapter snapshot until it
has authoritative evidence that the exact package was accepted.

If the POST acknowledgement is lost or ambiguous, the Client can query:

```text
GET /v1/rounds/{round_id}/submissions/{client_id}/receipt
```

The receipt lookup is authenticated and must match the exact round, Client, and
package hash. The Client commits only when the exact package is confirmed
accepted. Wrong hash/client/round or an unaccepted submission fails closed and
leaves the pending package intact.

This recovery path handles lost acknowledgements without turning retries into a
second logical submission.

## tmux split-round orchestration

`scripts/run_split_round_tmux.sh` orchestrates the role-separated real test while
keeping the remote Host/Coordinator and local Client visibly separate.

It creates panes/windows for:

```text
remote Host/Coordinator + remote GPU monitor
round runner / automatic round creation
local Client logs + local GPU monitor
Coordinator state / final evidence collector
SSH tunnel
```

The script uses an SSH ControlMaster so the user enters the remote password once.
The tunnel is used for Coordinator traffic; the remote Host remains bound to
loopback.

The generated cleanup script stops the tmux session, Client runtime, and SSH
ControlMaster while preserving run evidence.

### Final evidence behavior

Runner exit does not immediately freeze Coordinator evidence. The finalizer:

```text
records runner exit
      ↓
continues observing persisted Coordinator state
      ↓
COMPLETED / SKIPPED / ABORTED
      or bounded evidence timeout
      ↓
writes coordinator-final-state.json
      ↓
captures final Client health
      ↓
computes PASS / FAIL
```

Observation is evidence collection only. It does not retry a submission, call
`/sync`, mutate Client state, or otherwise recover a failed run automatically.

## Remote NVIDIA A40 environment

The remote Host/Coordinator path is designed around a writable runtime root such
as:

```text
/scratch/legalfedllm-test
```

with a source checkout under:

```text
/scratch/legalfedllm-test/work/LegalFedLLM
```

and a Python virtual environment under:

```text
/scratch/legalfedllm-test/.venv
```

The real Host/Coordinator services run directly from that virtual environment,
not through Docker. The Host binds to `127.0.0.1:8002`; the Coordinator binds to
`127.0.0.1:8000` and is exposed to the local Client only through SSH forwarding.

The verified environment uses an NVIDIA A40 with CUDA/BF16-capable PyTorch,
Transformers, PEFT, Accelerate, Triton, and safetensors.

### Remote container command limitations

The remote environment is intentionally treated as a constrained container, not
as a normal workstation/server installation.

In particular:

- `ss` is not available/supported in the container;
- do not infer port state from an empty unsupported port-inspection command;
- use `ps` to inspect the actual LegalFedLLM/Uvicorn processes;
- use direct `curl` requests to `/health` to confirm whether Host/Coordinator are
  alive or stopped;
- `lsof` or `fuser` may be useful when installed, but they should not replace the
  process/health check;
- `nvidia-smi` is available for GPU/VRAM monitoring; and
- use the writable `/scratch` runtime tree rather than assuming ordinary home or
  system paths are writable.

A practical cleanup check is:

```bash
ps -ef | grep -E \
  'run_host_stack.py|uvicorn.*host.main|uvicorn.*coordinator.main' \
  | grep -v grep || true
```

followed by direct health checks:

```bash
curl -fsS --max-time 2 http://127.0.0.1:8000/health || true
curl -fsS --max-time 2 http://127.0.0.1:8002/health || true
```

If stale Host/Coordinator processes exist, stop the `run_host_stack.py` parent
first and re-check the children before using stronger signals.

## Running the remote Host/Coordinator manually

From the remote repository root:

```bash
source /scratch/legalfedllm-test/.venv/bin/activate
set -a
. ./.env.host
set +a
python scripts/run_host_stack.py --env-file .env.host
```

Keep that process running while the SSH tunnel and local Client are active.

Useful health checks:

```bash
curl -fsS http://127.0.0.1:8002/health
curl -fsS http://127.0.0.1:8000/health
```

## Running a split Client manually

The local Client uses `compose.clients.yaml` and a role-specific environment
file. A typical workflow is:

```bash
docker compose \
  -p legalfedllm-split \
  --env-file .env.remote-client \
  -f compose.clients.yaml \
  --profile qwen \
  up -d --build
```

Then run the round driver from the repository environment:

```bash
python scripts/run_remote_round.py
```

For unattended testing, prefer `scripts/run_split_round_tmux.sh` so startup,
state observation, GPU monitoring, final evidence, and cleanup are collected
consistently.

## Tests and verification

### Ordinary repository suite

From the repository root:

```bash
python -m unittest discover -v
```

Some real-model tests are opt-in and require the pinned model/tokenizer artifacts
plus the expected CUDA/Transformers/PEFT environment. A missing optional ML
dependency should be distinguished from a source-code regression.

### Reliability and safety regressions

Focused model-free coverage includes:

```bash
python -m unittest -v \
  tests.test_package_safety \
  tests.test_submission_reconciliation \
  tests.test_split_round_tmux
```

These tests cover, among other things:

- lost-after-acceptance acknowledgement reconciliation;
- exact hash/client/round receipt matching;
- fail-closed unconfirmed submissions;
- responsive receipt/status handling while safety validation is still running;
- no premature accepted receipt during validation;
- linear-time safety frequency calculations; and
- terminal evidence collection after runner exit.

Shell syntax should also be checked with:

```bash
bash -n scripts/run_split_round_tmux.sh
```

### Real Qwen acceptance

Real Qwen tests are opt-in and require the configured tokenizer/model artifacts
and CUDA environment. They validate private Client LoRA training, real D^P
Knowledge Package generation, and package/reverse paths.

### Reverse Qwen acceptance

The reverse real-model path verifies that a Host-derived sparse training job can
train a new Qwen PEFT candidate and exercise the independent local adoption
logic. These tests qualify the implemented Client lifecycle; they do not qualify
a production malicious-update classifier.

### Full-D^P heterogeneous alignment validation

The deterministic alignment runner can be executed with:

```bash
python scripts/validate_fedmkt_alignment.py \
  --reference data/derived/gld2012/reference.jsonl \
  --output artifacts/fedmkt-alignment-validation.json \
  --mapping-cache artifacts/fedmkt-alignment-cache \
  --identity-dir artifacts/fedmkt-validation-identities \
  --maximum-sequence-length 4096 \
  --top-k 4
```

It records deterministic mapping/alignment identities and selection behavior;
it is not a model-quality benchmark.

## Service APIs

### Coordinator — port 8000

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Coordinator status/quorum policy |
| `GET` | `/v1/identity` | Coordinator + Host public identity |
| `POST` | `/v1/clients/register` | Register Client profile/public key |
| `POST` | `/v1/rounds` | Create/sign a round manifest |
| `GET` | `/v1/rounds/current` | Retrieve current manifest |
| `GET` | `/v1/rounds/{id}/manifest` | Retrieve one manifest |
| `GET` | `/v1/rounds/{id}/status` | Retrieve round state |
| `GET` | `/v1/rounds/{id}/reference-dataset` | Selected-Client D^P download |
| `POST` | `/v1/rounds/{id}/knowledge` | Upload signed package + artifact |
| `GET` | `/v1/rounds/{id}/submissions/{client_id}/receipt` | Authenticated exact accepted-submission receipt |
| `GET` | `/v1/rounds/{id}/safety` | Admin safety reports for the round |
| `GET` | `/v1/rounds/{id}/host-knowledge` | Download signed post-decision Host package |
| `POST` | `/v1/generate` | Proxy Host generation |

The receipt lookup requires the registration token and matching `X-Client-Id`.
The safety-report endpoint is admin-protected.

### Client — loopback port 8001

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Client state/backend status |
| `POST` | `/v1/register` | Register with Coordinator |
| `POST` | `/v1/local-train` | Local training outside a round |
| `POST` | `/v1/rounds/{id}/local-train` | Train against one signed round |
| `POST` | `/v1/participate` | Legacy current-round participation |
| `POST` | `/v1/rounds/{id}/participate` | Generate and submit exact round package |
| `POST` | `/v1/rounds/{id}/sync` | Verify Host package and execute reverse decision |
| `POST` | `/v1/generate` | Local mock/Ollama generation |
| `GET` | `/v1/ollama/models` | List configured Ollama models |
| `POST` | `/v1/ollama/inspect` | Inspect an Ollama model |

Client administrative endpoints require `X-Client-Admin-Token`.

### Host — private/loopback port 8002

The Host internal API is protected by `X-Internal-Token`. It exposes internal
identity, reference-data loading, reference knowledge, sparse training-job
intake, candidate training, candidate validation, post-decision knowledge, and
generation. In the remote topology it is loopback-only and is contacted by the
colocated Coordinator.

## Persistence and idempotence

LegalFedLLM does not require a database. Each role owns a filesystem state tree.
Important persisted Client state includes:

```text
identity and active adapter state
round-bound training records
PEFT checkpoints
verified D^P caches
pending and accepted Knowledge Packages
submission receipts
adapter snapshots
Host package cache
immutable reverse jobs and sparse artifacts
candidate/validation/safety/adoption records
```

Important Coordinator state includes:

```text
registered Client identities
signed manifests
accepted submissions
safety reports and trust history
round state
Host baseline package
integration audit
Host training job/receipt/result/validation decision
post-decision Host package
round audit events
```

Retries are exact where possible. A pending Client submission is revalidated
against its round, model profile, training record, checkpoint, and artifact.
Once accepted, the package/artifact/snapshot set is immutable.

## Security and privacy boundary

Implemented security controls include:

- Ed25519 identities and signatures;
- canonical JSON hashing/signing;
- exact artifact byte-size and SHA-256 binding;
- registered Client public keys;
- signed Coordinator manifests;
- model/tokenizer/adapter/reference-dataset/sample-order binding;
- nonce and package-hash replay protection;
- selected-Client authorization for D^P download;
- bounded multipart/package sizes;
- strict tensor names/dtypes/shapes/offsets/finite-value validation;
- gold-token/log-normalizer evidence consistency checks;
- temporary-file cleanup and immutable accepted storage;
- pre- and post-alignment package safety reports;
- trust-gated selection;
- hidden D^V validation and Host rollback;
- round/checkpoint provenance; and
- append-only audit records.

These controls do **not** provide confidentiality by themselves. Ed25519 proves
origin/integrity; it does not encrypt HTTP traffic or stored artifacts.

The current DP report enforces protocol/policy consistency only. Real model
training is ordinary LoRA training, not DP-SGD, and LegalFedLLM makes no formal
differential-privacy claim.

Knowledge/logit sharing can itself reveal information. The project should not be
described as formally private merely because raw Client examples and Client LoRA
weights remain local.

## Ollama boundary

Ollama is an optional **serving** boundary, not the federated training runtime.
Real training uses Transformers/PEFT.

The Qwen Client can serve through local Ollama (`qwen3:1.7b`). The Granite
compatibility profile can use `granite3.3:2b`. The pinned Mistral Nemo Host
currently supports `mock` serving only in the LegalFedLLM profile.

Promoting a PEFT training adapter does not currently export/import it into an
Ollama model automatically. Consequently `training_adapter_version` can advance
while `serving_adapter_version` remains unchanged. That is a known deployment
boundary and should not be interpreted as evidence that reverse training failed.

## FedMKT upstream/adaptation record

The optional machine-learning components under `shared/fedmkt_core/ml/` were
extracted and adapted from `FederatedAI/FATE-LLM`, package
`fate_llm.algo.fedmkt`, at commit:

```text
0c63377e468f0f62a9bdf5fb32424688b9478553
```

LegalFedLLM retains the reviewed DTW/minimum-CE behavior while replacing FATE
communication/orchestration with its own protocol, HTTP, persistence, and
security layers. It also uses answer-only supervision, deterministic
demand-driven vocabulary mapping, and operational sparse targets.

See:

- `shared/fedmkt_core/UPSTREAM.md`
- `shared/fedmkt_core/PARITY.md`
- `shared/fedmkt_core/LICENSE`
- `THIRD_PARTY_NOTICES.md`

The upstream FATE Context, Guest/Host/Arbiter channels, FATE-Flow, and parameter
aggregation wrappers are not part of LegalFedLLM.

## Current limitations

The current repository does **not** establish:

- forward Client-to-Host teaching in the verified real experiment: Qwen was
  selected on `0 / 565` forward samples;
- a real multi-Client federated round;
- mixed Qwen + Granite per-Client alignment identities inside one live manifest;
- a production-calibrated malicious-Knowledge-Package detector;
- an independently validated production SafeFed-style Qwen LoRA probe;
- formal DP-SGD or differential-privacy accounting;
- HTTPS/mTLS, production Client enrollment/certificate provisioning, or encrypted
  artifact storage;
- automatic PEFT-adapter publication into Ollama;
- real Mistral Nemo serving through the current Host profile;
- stable thesis measurements for end-to-end wall time, peak RAM/VRAM,
  communication cost, and scaling across multiple real Clients; or
- a graphical end-user application.

D^V validation is a strong held-out outcome gate, but it is not a proof against
all targeted/backdoor behavior outside D^V coverage. Likewise, a low or accepted
PoC trust score is not a production security certification.

## Next experimental work

The main open experimental questions are:

1. run a real multi-Client round under the normal majority/minimum-2 quorum;
2. measure scenarios in which an eligible Client actually wins some forward
   DualMinCE samples, and report teacher-selection counts explicitly;
3. independently train/calibrate the Client safety probe and evaluate malicious
   package/adapter cases rather than relying on protocol fixtures;
4. collect stable wall-time, RAM/VRAM, communication-volume, and adapter-size
   measurements for thesis experiments; and
5. decide whether automatic PEFT → serving-model publication belongs in the PoC
   scope.

These are experiment/deployment boundaries. They should not be documented as
completed until measured in the agreed authoritative environment.

## Accurate project claim

A defensible current summary is:

> LegalFedLLM implements a protocol-first heterogeneous federated language-model
> proof of concept in which model-native LoRA weights and private Client examples
> remain local while signed behavioral Knowledge Packages are exchanged over a
> common reference dataset. A fresh real cross-machine Qwen3 1.7B → Mistral Nemo
> → Qwen run completed package verification, SafeFed-inspired screening, DTW
> alignment, DualMinCE selection, Host candidate training, hidden D^V promotion,
> signed Host publication, automatic Client synchronization, and reverse Qwen
> candidate adoption without manual recovery. In that experiment the Host
> self-teacher won all 565 forward samples, so the measured Host validation
> improvement cannot be attributed to Qwen-to-Host knowledge transfer; the
> reverse path did select the Host on 508 samples.

The repository is beyond a mock protocol demonstration, but it remains a
research proof of concept rather than a production federated-learning system.
