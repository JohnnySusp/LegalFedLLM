from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from shared.knowledge_transport import (
    KnowledgeTransportError,
    KnowledgeTransportTooLarge,
    knowledge_transfer_response,
    receive_knowledge_transfer,
)


class Metadata(BaseModel):
    package_hash: str
    sender_id: str


async def chunks(value: bytes, width: int = 11):
    for index in range(0, len(value), width):
        yield value[index : index + width]


def multipart_body(
    parts: list[tuple[str, bytes, str]],
    *,
    boundary: str = "test-boundary",
    close: bool = True,
) -> bytes:
    body = bytearray()
    for name, value, media_type in parts:
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{name}.bin"\r\n'
                f"Content-Type: {media_type}\r\n\r\n"
            ).encode("ascii")
        )
        body.extend(value)
        body.extend(b"\r\n")
    if close:
        body.extend(f"--{boundary}--\r\n".encode("ascii"))
    return bytes(body)


class KnowledgeTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_streamed_multipart_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            source.write_bytes(b"safetensors-bytes" * 100)
            metadata = Metadata(
                package_hash="a" * 64,
                sender_id="client-a",
            )
            response = knowledge_transfer_response(
                metadata=metadata,
                artifact_path=source,
                metadata_part_name="package",
            )
            body = b"".join(
                [chunk async for chunk in response.body_iterator]
            )
            target = root / "received.safetensors"
            received = await receive_knowledge_transfer(
                content_type=response.headers["content-type"],
                chunks=chunks(body, 7),
                artifact_path=target,
                metadata_part_name="package",
                maximum_content_bytes=4096,
            )

            self.assertEqual(received.metadata, metadata.model_dump())
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertEqual(
                received.content_size,
                received.metadata_size + source.stat().st_size,
            )

    async def test_missing_and_duplicate_parts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content_type = "multipart/form-data; boundary=test-boundary"
            missing = multipart_body(
                [("package", b"{}", "application/json")]
            )
            target = root / "missing.safetensors"
            with self.assertRaisesRegex(
                KnowledgeTransportError,
                "missing a required part",
            ):
                await receive_knowledge_transfer(
                    content_type=content_type,
                    chunks=chunks(missing),
                    artifact_path=target,
                    metadata_part_name="package",
                    maximum_content_bytes=4096,
                )
            self.assertFalse(target.exists())

            duplicate = multipart_body(
                [
                    ("package", b"{}", "application/json"),
                    ("artifact", b"first", "application/octet-stream"),
                    ("artifact", b"second", "application/octet-stream"),
                ]
            )
            with self.assertRaisesRegex(
                KnowledgeTransportError,
                "duplicate part",
            ):
                await receive_knowledge_transfer(
                    content_type=content_type,
                    chunks=chunks(duplicate),
                    artifact_path=target,
                    metadata_part_name="package",
                    maximum_content_bytes=4096,
                )
            self.assertFalse(target.exists())

    async def test_oversized_and_truncated_transfers_are_cleaned_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content_type = "multipart/form-data; boundary=test-boundary"
            oversized = multipart_body(
                [
                    ("package", b"{}", "application/json"),
                    ("artifact", b"x" * 100, "application/octet-stream"),
                ]
            )
            target = root / "oversized.safetensors"
            with self.assertRaises(KnowledgeTransportTooLarge):
                await receive_knowledge_transfer(
                    content_type=content_type,
                    chunks=chunks(oversized),
                    artifact_path=target,
                    metadata_part_name="package",
                    maximum_content_bytes=50,
                )
            self.assertFalse(target.exists())

            truncated = multipart_body(
                [
                    ("package", b"{}", "application/json"),
                    ("artifact", b"artifact", "application/octet-stream"),
                ],
                close=False,
            )
            with self.assertRaisesRegex(
                KnowledgeTransportError,
                "incomplete",
            ):
                await receive_knowledge_transfer(
                    content_type=content_type,
                    chunks=chunks(truncated),
                    artifact_path=target,
                    metadata_part_name="package",
                    maximum_content_bytes=4096,
                )
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
