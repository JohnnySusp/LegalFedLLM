from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.model_profiles import (
    GRANITE_3_3_2B_CLIENT_PROFILE_ID,
    QWEN_PROFILE_ID,
)
from coordinator.quorum import TrustedClientQuorumPolicy
from shared.env_bootstrap import load_env_file
from shared.prompt import PROMPT_TEMPLATE


_UNSAFE_SECRET_PREFIXES = (
    "development-",
    "placeholder-",
    "replace-this-",
)


@dataclass(frozen=True, slots=True)
class RoundClientSlot:
    name: str
    client_id: str
    profile_id: str


def _required_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith(_UNSAFE_SECRET_PREFIXES):
        raise ValueError(f"{name} must contain a non-placeholder secret")
    return value


def configured_round_slots() -> dict[str, RoundClientSlot]:
    return {
        "client-1": RoundClientSlot(
            name="client-1",
            client_id=os.getenv("QWEN_CLIENT_ID", "legal-client-1").strip(),
            profile_id=QWEN_PROFILE_ID,
        ),
        "client-2": RoundClientSlot(
            name="client-2",
            client_id=os.getenv("GRANITE_CLIENT_ID", "legal-client-2").strip(),
            profile_id=GRANITE_3_3_2B_CLIENT_PROFILE_ID,
        ),
    }


def selected_round_slots() -> list[RoundClientSlot]:
    available = configured_round_slots()
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
    return [available[name] for name in names]


PRIVATE_LABEL_FORMAT = "chat_sft_answer_only_v1"


def build_round_request(
    *,
    slots: list[RoundClientSlot],
    expected_quorum: int,
) -> dict[str, Any]:
    return {
        "selected_client_ids": [slot.client_id for slot in slots],
        "trusted_client_quorum": expected_quorum,
        "prompt_template": PROMPT_TEMPLATE,
        "label_format": PRIVATE_LABEL_FORMAT,
        "top_k": int(os.getenv("ROUND_TOP_K", "4")),
        "maximum_sequence_length": int(
            os.getenv("ROUND_MAXIMUM_SEQUENCE_LENGTH", "4096")
        ),
        "truncation_policy": "reject",
        "training_epochs": int(os.getenv("ROUND_PRIVATE_TRAINING_EPOCHS", "1")),
        "host_public_data_epochs": int(
            os.getenv("ROUND_HOST_PUBLIC_DATA_EPOCHS", "5")
        ),
        "submission_window_seconds": int(
            os.getenv("ROUND_SUBMISSION_WINDOW_SECONDS", "21600")
        ),
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a real split-topology round from the Host/Coordinator side."
    )
    parser.add_argument(
        "--env-file",
        default=str(ROOT / ".env"),
        help="private Host environment file created by scripts/bootstrap.py host",
    )
    args = parser.parse_args()
    load_env_file(args.env_file)

    import httpx

    coordinator_url = os.getenv(
        "ROUND_COORDINATOR_URL",
        "http://127.0.0.1:8000",
    ).rstrip("/")
    admin_token = _required_secret("ADMIN_TOKEN")
    slots = selected_round_slots()
    override_text = os.getenv(
        "COORDINATOR_TRUSTED_CLIENT_QUORUM_OVERRIDE",
        "",
    ).strip()
    policy = TrustedClientQuorumPolicy(
        minimum=int(os.getenv("COORDINATOR_MINIMUM_TRUSTED_CLIENT_QUORUM", "2")),
        override=int(override_text) if override_text else None,
    )
    expected_quorum = policy.resolve(len(slots))

    round_request = build_round_request(
        slots=slots,
        expected_quorum=expected_quorum,
    )

    timeout = float(os.getenv("ROUND_TIMEOUT_SECONDS", "21600"))
    with httpx.Client(timeout=timeout) as client:
        health = _show("Coordinator health", client.get(f"{coordinator_url}/health"))
        if health.get("quorum_policy") != "majority":
            raise RuntimeError("Coordinator is not using the majority quorum policy")
        manifest = _show(
            "round creation",
            client.post(
                f"{coordinator_url}/v1/rounds",
                headers={"X-Admin-Token": admin_token},
                json=round_request,
            ),
        )

    if manifest.get("trusted_client_quorum") != expected_quorum:
        raise RuntimeError("Coordinator resolved an unexpected trusted quorum")

    print(
        json.dumps(
            {
                "round_id": manifest["round_id"],
                "selected_client_ids": manifest["selected_client_ids"],
                "trusted_client_quorum": manifest["trusted_client_quorum"],
                "host_public_data_epochs": manifest["host_public_data_epochs"],
                "selected_client_alignment_profiles": manifest[
                    "selected_client_alignment_profiles"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
