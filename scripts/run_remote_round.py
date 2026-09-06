from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.env_bootstrap import load_env_file


_UNSAFE_SECRET_PREFIXES = (
    "development-",
    "placeholder-",
    "replace-this-",
)


@dataclass(frozen=True, slots=True)
class ClientSlot:
    name: str
    expected_client_id: str
    url: str
    admin_token: str


def configured_client_slots() -> dict[str, ClientSlot]:
    return {
        "client-1": ClientSlot(
            name="client-1",
            expected_client_id=os.getenv(
                "QWEN_CLIENT_ID",
                "legal-client-1",
            ).strip(),
            url=os.getenv(
                "QWEN_CLIENT_URL",
                "http://127.0.0.1:8001",
            ).rstrip("/"),
            admin_token=os.getenv("QWEN_CLIENT_ADMIN_TOKEN", "").strip(),
        ),
        "client-2": ClientSlot(
            name="client-2",
            expected_client_id=os.getenv(
                "GRANITE_CLIENT_ID",
                "legal-client-2",
            ).strip(),
            url=os.getenv(
                "GRANITE_CLIENT_URL",
                "http://127.0.0.1:8002",
            ).rstrip("/"),
            admin_token=os.getenv("GRANITE_CLIENT_ADMIN_TOKEN", "").strip(),
        ),
    }


def enabled_client_slots() -> list[ClientSlot]:
    available = configured_client_slots()
    names = [
        item.strip()
        for item in os.getenv("ROUND_CLIENT_SLOTS", "client-1").split(",")
        if item.strip()
    ]
    if not names:
        raise ValueError("ROUND_CLIENT_SLOTS must select at least one Client")
    if len(names) != len(set(names)):
        raise ValueError("ROUND_CLIENT_SLOTS must not contain duplicates")
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"unknown Client slots: {', '.join(unknown)}")
    selected = [available[name] for name in names]
    missing_tokens = [
        slot.name
        for slot in selected
        if not slot.admin_token
        or slot.admin_token.startswith(_UNSAFE_SECRET_PREFIXES)
    ]
    if missing_tokens:
        raise ValueError(
            "missing Client admin token for: " + ", ".join(missing_tokens)
        )
    return selected


def _show(label: str, response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {"text": response.text[:1000]}
    if response.status_code >= 400:
        raise RuntimeError(
            f"{label} returned {response.status_code}: "
            f"{json.dumps(payload, sort_keys=True)}"
        )
    print(f"{label}: {json.dumps(payload, sort_keys=True)}", flush=True)
    return payload


def _client_headers(slot: ClientSlot) -> dict[str, str]:
    return {"X-Client-Admin-Token": slot.admin_token}


def _wait_for_round(
    *,
    client: Any,
    coordinator_url: str,
    selected_client_ids: list[str],
    deadline: float,
    poll_seconds: float,
) -> dict[str, Any]:
    expected = set(selected_client_ids)
    while True:
        response = client.get(f"{coordinator_url}/v1/rounds/current")
        if response.status_code == 200:
            manifest = response.json()
            if set(manifest.get("selected_client_ids", [])) == expected:
                status_response = client.get(
                    f"{coordinator_url}/v1/rounds/{manifest['round_id']}/status"
                )
                if status_response.status_code == 200:
                    state = status_response.json().get("state")
                    if state == "COLLECTING":
                        print(
                            f"active round: {json.dumps(manifest, sort_keys=True)}",
                            flush=True,
                        )
                        return manifest
        elif response.status_code not in {404, 409}:
            _show("current round", response)

        if time.monotonic() >= deadline:
            raise TimeoutError("no matching COLLECTING round became available")
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Register local Clients, wait for the Host-side operator to create a round, "
            "then execute local train, participate and reverse sync."
        )
    )
    parser.add_argument(
        "--env-file",
        default=str(ROOT / ".env"),
        help="private Client environment file created by scripts/bootstrap.py client",
    )
    args = parser.parse_args()
    load_env_file(args.env_file)

    import httpx

    coordinator_url = os.getenv(
        "ROUND_COORDINATOR_URL",
        "http://127.0.0.1:8000",
    ).rstrip("/")
    requested_slots = enabled_client_slots()
    request_timeout = float(os.getenv("ROUND_TIMEOUT_SECONDS", "21600"))
    poll_seconds = float(os.getenv("ROUND_POLL_INTERVAL_SECONDS", "5"))

    connected: list[tuple[ClientSlot, dict[str, Any]]] = []
    with httpx.Client(timeout=30) as client:
        _show("Coordinator health", client.get(f"{coordinator_url}/health"))
        for slot in requested_slots:
            try:
                health_response = client.get(f"{slot.url}/health")
            except httpx.HTTPError as exc:
                print(f"{slot.name} is not reachable: {exc}", flush=True)
                continue
            health = _show(f"{slot.name} health", health_response)
            if health.get("client_id") != slot.expected_client_id:
                raise RuntimeError(
                    f"{slot.name} returned Client ID {health.get('client_id')!r}; "
                    f"expected {slot.expected_client_id!r}"
                )
            registration = _show(
                f"{slot.name} registration",
                client.post(
                    f"{slot.url}/v1/register",
                    headers=_client_headers(slot),
                ),
            )
            if registration.get("client_id") != slot.expected_client_id:
                raise RuntimeError(f"{slot.name} registered another Client ID")
            connected.append((slot, registration))

    if not connected:
        raise RuntimeError("none of the requested Clients is connected")

    selected_client_ids = [slot.expected_client_id for slot, _ in connected]
    print(
        "Clients are registered. Create the round from the Host/Coordinator side now.",
        flush=True,
    )

    deadline = time.monotonic() + request_timeout
    with httpx.Client(timeout=request_timeout) as client:
        manifest = _wait_for_round(
            client=client,
            coordinator_url=coordinator_url,
            selected_client_ids=selected_client_ids,
            deadline=deadline,
            poll_seconds=poll_seconds,
        )
        round_id = str(manifest["round_id"])

        for slot, _ in connected:
            _show(
                f"{slot.name} local training",
                client.post(
                    f"{slot.url}/v1/rounds/{round_id}/local-train",
                    headers=_client_headers(slot),
                ),
            )
            _show(
                f"{slot.name} participation",
                client.post(
                    f"{slot.url}/v1/rounds/{round_id}/participate",
                    headers=_client_headers(slot),
                ),
            )

        while True:
            state = _show(
                "round status",
                client.get(f"{coordinator_url}/v1/rounds/{round_id}/status"),
            )
            if state.get("state") in {"COMPLETED", "SKIPPED", "ABORTED"}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("round did not reach a terminal state")
            time.sleep(poll_seconds)

        if state.get("state") != "COMPLETED":
            raise RuntimeError(
                f"round ended in {state.get('state')}: {state.get('message')}"
            )

        for slot, _ in connected:
            _show(
                f"{slot.name} synchronization",
                client.post(
                    f"{slot.url}/v1/rounds/{round_id}/sync",
                    headers=_client_headers(slot),
                ),
            )

    print(
        json.dumps(
            {
                "round_id": round_id,
                "selected_client_ids": selected_client_ids,
                "trusted_client_quorum": manifest["trusted_client_quorum"],
                "state": state,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
