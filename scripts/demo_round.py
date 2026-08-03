from __future__ import annotations

import json
import os

import httpx


COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://localhost:8000")
CLIENT_URL = os.getenv("CLIENT_URL", "http://localhost:8001")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "replace-this-admin-token")
CLIENT_ADMIN_TOKEN = os.getenv(
    "CLIENT_ADMIN_TOKEN",
    "replace-this-client-admin-token",
)

CLIENT_ADMIN_HEADERS = {
    "X-Client-Admin-Token": CLIENT_ADMIN_TOKEN
}

def show(label: str, response: httpx.Response) -> dict:
    print(f"\n{label}: {response.status_code}")
    try:
        payload = response.json()
        print(json.dumps(payload, indent=2, sort_keys=True))
    except ValueError:
        print(response.text)
        response.raise_for_status()
        return {}
    response.raise_for_status()
    return payload


def main() -> None:
    with httpx.Client(timeout=60.0) as client:
        show(
            "register Client",
            client.post(
                f"{CLIENT_URL}/v1/register",
                headers=CLIENT_ADMIN_HEADERS,
            ),
        )
        manifest = show(
            "create round",
            client.post(
                f"{COORDINATOR_URL}/v1/rounds",
                headers={"X-Admin-Token": ADMIN_TOKEN},
                json={
                    "selected_client_ids": ["legal-client-1"],
                    "trusted_client_quorum": 1,
                    "reference_dataset_id": "mock-legal-reference-v1",
                    "reference_dataset_hash": "3a26263fdd6ae878071e2f52990ba0b0c6f7213bb3564b35bb1fa8fd174653b9",
                    "sample_ids": ["contract-001", "contract-002", "contract-003"],
                    "prompt_template": "Question: {question}\\nAnswer: {answer}",
                    "top_k": 4,
                    "maximum_sequence_length": 64,
                    "submission_window_seconds": 300,
                },
            ),
        )
        round_id = manifest["round_id"]
        show(
            "local Client training",
            client.post(
                f"{CLIENT_URL}/v1/local-train",
                headers=CLIENT_ADMIN_HEADERS,
                json={
                    "examples": [
                        "Private local example A",
                        "Private local example B",
                    ]
                },
            ),
        )
        show(
            "participate",
            client.post(
                f"{CLIENT_URL}/v1/participate",
                headers=CLIENT_ADMIN_HEADERS,
            ),
        )
        show(
            "round status",
            client.get(f"{COORDINATOR_URL}/v1/rounds/{round_id}/status"),
        )
        show(
            "Client sync",
            client.post(
                f"{CLIENT_URL}/v1/rounds/{round_id}/sync",
                headers=CLIENT_ADMIN_HEADERS,
            ),
        )


if __name__ == "__main__":
    main()
