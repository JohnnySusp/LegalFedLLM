# LegalFedLLM

LegalFedLLM is a mock-first proof of concept for exchanging federated LoRA adapter updates between a legal-domain Client model and a Central/Host model. The current milestone demonstrates the service and adapter boundaries without downloading a model, requiring a GPU, starting Flower, or connecting to a blockchain.

## Implemented flow

1. The Client fetches the current global adapter from the Host.
2. The Client installs it using the `replace` policy.
3. The mock Client backend derives a deterministic adapter delta from local examples.
4. The Client submits only the delta, client/round metadata, sample count, parent adapter hash, and update hash.
5. The Host verifies the hash, round lineage, adapter specification, tensor names, and tensor lengths.
6. The Host applies the single-client update and returns a new hashed global adapter.
7. The Client installs the returned adapter.

The raw examples are not part of the submitted update. The mock tensor values and hashes prove the data flow; they do not represent useful model training.

## Project layout

```text
LegalFedLLM/
├── host/               Host HTTP API and Central model runtime wrapper
├── client/             Client HTTP API, Host gateway, and local model wrapper
├── shared/adapter.py   Adapter snapshot/update contracts and verification
├── tests/test_round.py Mock round and API tests
├── compose.yaml
└── requirements.txt
```

Both runtime wrappers use a small backend interface. A real Transformers/PEFT implementation can replace the mock backends later while keeping the HTTP and adapter workflow stable.

The mock Host and Client deliberately share one adapter specification. If later experiments use different model architectures or dimensions for the SLM and Central LLM, direct LoRA exchange will not be compatible and will require an explicit projection, distillation, or other alignment stage.

## Run with Docker Compose

Docker and Docker Compose are the only requirements for the default path.

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
```

Check both services:

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:8001/health | python3 -m json.tool
```

Run one federated adapter round. The example text is accepted by the Client service and is not included in the Client-to-Host update envelope.

```bash
curl -s -X POST http://localhost:8001/v1/rounds \
  -H 'Content-Type: application/json' \
  -d '{"examples":["Local legal example A","Local legal example B"]}' \
  | python3 -m json.tool
```

Inspect the new global adapter and exercise both model wrappers:

```bash
curl -s http://localhost:8000/v1/adapters/global | python3 -m json.tool

curl -s -X POST http://localhost:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain the Host boundary."}' \
  | python3 -m json.tool

curl -s -X POST http://localhost:8001/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain the Client boundary."}' \
  | python3 -m json.tool
```

Stop the services with `docker compose down`.

## Run locally

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
```

In separate terminals, start the services with:

```bash
uvicorn host.main:app --host 0.0.0.0 --port 8000
HOST_URL=http://localhost:8000 uvicorn client.main:app --host 0.0.0.0 --port 8001
```

## HTTP surface

| Service | Method and path | Responsibility |
| --- | --- | --- |
| Host | `GET /health` | Host/backend status and global adapter version |
| Host | `GET /v1/adapters/global` | Return the current global adapter snapshot |
| Host | `POST /v1/adapter-updates` | Validate and apply one Client delta |
| Host | `POST /v1/generate` | Exercise the Central model wrapper |
| Client | `GET /health` | Client/backend status and local adapter version |
| Client | `POST /v1/rounds` | Execute one complete mock Host/Client round |
| Client | `POST /v1/generate` | Exercise the local model wrapper |

## Prototype limits

This is a one-Client, in-memory adapter exchange. Restarting a service resets its adapter. It does not yet implement multi-client aggregation, persistent checkpoints, real LoRA tensors, Transformers/PEFT training, authentication, signatures, encryption, differential privacy, secure aggregation, safety scoring, blockchain commitments, or audit storage. A SHA-256 envelope detects accidental/tampered payload changes but is not an authenticated client signature.

The mock generator is not a legal model and its output is not legal advice.

## Reference influences

The local research repositories were inspected rather than vendored. The adapter-only `get`/`set` and Client fit boundary follows the pattern used by FlowerTune. Safe-FedLLM informed the explicit delta boundary and future safety-check seam. FedJudge supports the legal federated-model scenario and adapter-only communication choice. GridSense informed the compact layout, Compose startup, environment example, repeatable tests, and explicit prototype limits. No source code from those repositories is copied into this skeleton.
