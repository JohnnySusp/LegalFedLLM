from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.env_bootstrap import load_env_file


_UNSAFE_SECRET_PREFIXES = (
    "development-",
    "placeholder-",
    "replace-this-",
)


def _required_admin_token() -> str:
    value = os.getenv("ADMIN_TOKEN", "").strip()
    if not value or value.startswith(_UNSAFE_SECRET_PREFIXES):
        raise ValueError("ADMIN_TOKEN must contain a non-placeholder secret")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue one single-use LegalFedLLM Client enrollment token."
    )
    parser.add_argument(
        "--env-file",
        default=str(ROOT / ".env"),
        help="Host environment file containing ADMIN_TOKEN",
    )
    parser.add_argument(
        "--coordinator-url",
        default="http://127.0.0.1:8000",
        help="Coordinator base URL",
    )
    parser.add_argument(
        "--token-only",
        action="store_true",
        help="print only the newly issued token",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    admin_token = _required_admin_token()

    import httpx

    with httpx.Client(timeout=30) as client:
        response = client.post(
            f"{args.coordinator_url.rstrip('/')}/v1/enrollment-tokens",
            headers={"X-Admin-Token": admin_token},
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Coordinator returned {response.status_code}: {response.text[:500]}"
        )
    payload = response.json()
    token = str(payload.get("token", "")).strip()
    if not token:
        raise RuntimeError("Coordinator returned no enrollment token")

    if args.token_only:
        print(token)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
