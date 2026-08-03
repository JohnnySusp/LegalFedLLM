from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(value: Any) -> str:
    payload = value if isinstance(value, bytes) else canonical_json_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


class Ed25519Identity:
    def __init__(self, private_key: Ed25519PrivateKey):
        self._private_key = private_key

    @classmethod
    def load_or_create(cls, path: str | Path) -> "Ed25519Identity":
        key_path = Path(path)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        if key_path.exists():
            private_key = serialization.load_pem_private_key(
                key_path.read_bytes(), password=None
            )
            if not isinstance(private_key, Ed25519PrivateKey):
                raise ValueError("identity key is not Ed25519")
            return cls(private_key)

        private_key = Ed25519PrivateKey.generate()
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        temporary = key_path.with_suffix(key_path.suffix + ".tmp")
        temporary.write_bytes(pem)
        temporary.replace(key_path)
        try:
            key_path.chmod(0o600)
        except OSError:
            pass
        return cls(private_key)

    @property
    def public_key_b64(self) -> str:
        raw = self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return _b64encode(raw)

    def sign_bytes(self, payload: bytes) -> str:
        return _b64encode(self._private_key.sign(payload))

    def sign_json(self, payload: Any) -> str:
        return self.sign_bytes(canonical_json_bytes(payload))


def verify_bytes(public_key_b64: str, payload: bytes, signature_b64: str) -> bool:
    try:
        key = Ed25519PublicKey.from_public_bytes(_b64decode(public_key_b64))
        key.verify(_b64decode(signature_b64), payload)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def verify_json(public_key_b64: str, payload: Any, signature_b64: str) -> bool:
    return verify_bytes(public_key_b64, canonical_json_bytes(payload), signature_b64)
