from __future__ import annotations

import json
import os
import secrets
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from typing import Any, BinaryIO

from pydantic import BaseModel
from python_multipart import MultipartParser
from starlette.responses import StreamingResponse

from shared.crypto import canonical_json_bytes

ARTIFACT_PART_NAME = "artifact"
MAXIMUM_METADATA_BYTES = 1024 * 1024
MAXIMUM_MULTIPART_OVERHEAD_BYTES = 64 * 1024


class KnowledgeTransportError(ValueError):
    pass


class KnowledgeTransportTooLarge(KnowledgeTransportError):
    pass


@dataclass(frozen=True)
class ReceivedKnowledgeTransfer:
    metadata: dict[str, Any]
    metadata_size: int
    artifact_path: Path
    artifact_size: int

    @property
    def content_size(self) -> int:
        return self.metadata_size + self.artifact_size


def _header_parameter(value: str, name: str) -> str | None:
    message = Message()
    message["content-disposition"] = value
    parameter = message.get_param(name, header="content-disposition")
    return parameter if isinstance(parameter, str) else None


def _multipart_boundary(content_type: str) -> str:
    message = Message()
    message["content-type"] = content_type
    if message.get_content_type() != "multipart/form-data":
        raise KnowledgeTransportError("Knowledge Package must use multipart/form-data")
    boundary = message.get_param("boundary", header="content-type")
    if not isinstance(boundary, str) or not boundary or len(boundary) > 200:
        raise KnowledgeTransportError("multipart boundary is missing or invalid")
    return boundary


async def receive_knowledge_transfer(
    *,
    content_type: str,
    chunks: AsyncIterable[bytes],
    artifact_path: str | Path,
    metadata_part_name: str,
    maximum_content_bytes: int,
) -> ReceivedKnowledgeTransfer:
    if maximum_content_bytes < 1:
        raise ValueError("maximum_content_bytes must be positive")

    target = Path(artifact_path)
    if target.exists():
        raise KnowledgeTransportError("artifact destination already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.part"
    )

    metadata = bytearray()
    artifact_size = 0
    content_size = 0
    raw_size = 0
    current_header_name = bytearray()
    current_header_value = bytearray()
    headers: dict[str, str] = {}
    current_part: str | None = None
    artifact_handle: BinaryIO | None = None
    seen_parts: set[str] = set()
    ended = False

    def on_part_begin() -> None:
        nonlocal headers, current_part
        headers = {}
        current_part = None

    def on_header_begin() -> None:
        current_header_name.clear()
        current_header_value.clear()

    def on_header_field(data: bytes, start: int, end: int) -> None:
        current_header_name.extend(data[start:end])

    def on_header_value(data: bytes, start: int, end: int) -> None:
        current_header_value.extend(data[start:end])

    def on_header_end() -> None:
        try:
            name = current_header_name.decode("ascii").lower()
            value = current_header_value.decode("utf-8")
        except UnicodeError as exc:
            raise KnowledgeTransportError("multipart header is not valid text") from exc
        if name in headers:
            raise KnowledgeTransportError("duplicate multipart header")
        headers[name] = value

    def on_headers_finished() -> None:
        nonlocal current_part, artifact_handle
        disposition = headers.get("content-disposition")
        if disposition is None:
            raise KnowledgeTransportError("multipart part lacks Content-Disposition")
        part_name = _header_parameter(disposition, "name")
        if part_name not in {metadata_part_name, ARTIFACT_PART_NAME}:
            raise KnowledgeTransportError("multipart contains an unknown part")
        if part_name in seen_parts:
            raise KnowledgeTransportError("multipart contains a duplicate part")
        current_part = part_name
        if current_part == ARTIFACT_PART_NAME:
            artifact_handle = temporary.open("xb")

    def on_part_data(data: bytes, start: int, end: int) -> None:
        nonlocal artifact_size, content_size
        if current_part is None:
            raise KnowledgeTransportError("multipart part data has no part identity")
        value = data[start:end]
        content_size += len(value)
        if content_size > maximum_content_bytes:
            raise KnowledgeTransportTooLarge(
                "Knowledge Package exceeds its size limit"
            )
        if current_part == metadata_part_name:
            if len(metadata) + len(value) > MAXIMUM_METADATA_BYTES:
                raise KnowledgeTransportTooLarge(
                    "Knowledge Package metadata is too large"
                )
            metadata.extend(value)
        else:
            if artifact_handle is None:
                raise KnowledgeTransportError("artifact part is not writable")
            artifact_handle.write(value)
            artifact_size += len(value)

    def on_part_end() -> None:
        nonlocal artifact_handle
        if current_part is None:
            raise KnowledgeTransportError("multipart part ended without an identity")
        if current_part == ARTIFACT_PART_NAME and artifact_handle is not None:
            artifact_handle.close()
            artifact_handle = None
        seen_parts.add(current_part)

    def on_end() -> None:
        nonlocal ended
        ended = True

    parser = MultipartParser(
        _multipart_boundary(content_type),
        {
            "on_part_begin": on_part_begin,
            "on_header_begin": on_header_begin,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
            "on_part_data": on_part_data,
            "on_part_end": on_part_end,
            "on_end": on_end,
        },
        max_header_count=4,
        max_header_size=4096,
    )

    try:
        raw_limit = maximum_content_bytes + MAXIMUM_MULTIPART_OVERHEAD_BYTES
        async for chunk in chunks:
            raw_size += len(chunk)
            if raw_size > raw_limit:
                raise KnowledgeTransportTooLarge(
                    "multipart transfer exceeds its size limit"
                )
            parser.write(chunk)
        parser.finalize()
        if not ended:
            raise KnowledgeTransportError("multipart transfer is incomplete")
        if seen_parts != {metadata_part_name, ARTIFACT_PART_NAME}:
            raise KnowledgeTransportError("multipart transfer is missing a required part")
        if not metadata:
            raise KnowledgeTransportError("Knowledge Package metadata is empty")
        if artifact_size < 1:
            raise KnowledgeTransportError("Knowledge Package artifact is empty")
        try:
            payload = json.loads(metadata)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise KnowledgeTransportError("Knowledge Package metadata is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise KnowledgeTransportError("Knowledge Package metadata must be an object")
        temporary.replace(target)
        return ReceivedKnowledgeTransfer(
            metadata=payload,
            metadata_size=len(metadata),
            artifact_path=target,
            artifact_size=artifact_size,
        )
    except Exception:
        if artifact_handle is not None:
            artifact_handle.close()
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise


def knowledge_transfer_response(
    *,
    metadata: BaseModel,
    artifact_path: str | Path,
    metadata_part_name: str,
) -> StreamingResponse:
    artifact = Path(artifact_path)
    metadata_bytes = canonical_json_bytes(metadata.model_dump(mode="json"))
    boundary = f"legalfedllm-{secrets.token_hex(24)}"
    opening = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{metadata_part_name}"; '
        f'filename="{metadata_part_name}.json"\r\n'
        "Content-Type: application/json\r\n\r\n"
    ).encode("ascii")
    middle = (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{ARTIFACT_PART_NAME}"; '
        'filename="knowledge.safetensors"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("ascii")
    closing = f"\r\n--{boundary}--\r\n".encode("ascii")

    async def body() -> AsyncIterator[bytes]:
        yield opening
        yield metadata_bytes
        yield middle
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                yield chunk
        yield closing

    content_length = (
        len(opening)
        + len(metadata_bytes)
        + len(middle)
        + artifact.stat().st_size
        + len(closing)
    )
    return StreamingResponse(
        body(),
        media_type=f"multipart/form-data; boundary={boundary}",
        headers={"Content-Length": str(content_length)},
    )
