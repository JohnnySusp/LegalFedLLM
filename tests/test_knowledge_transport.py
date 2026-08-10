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

            duplicates = {
                "package": [
                    ("package", b"{}", "application/json"),
                    ("package", b"{}", "application/json"),
                    ("artifact", b"artifact", "application/octet-stream"),
                ],
                "artifact": [
                    ("package", b"{}", "application/json"),
                    ("artifact", b"first", "application/octet-stream"),
                    ("artifact", b"second", "application/octet-stream"),
                ],
            }
            for part_name, parts in duplicates.items():
                with self.subTest(part_name=part_name):
                    duplicate = multipart_body(parts)
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

    async def test_malformed_metadata_is_rejected_and_cleaned_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content_type = "multipart/form-data; boundary=test-boundary"
            payloads = {
                "invalid JSON": b'{"package_hash":',
                "must be an object": b"[]",
                "metadata is empty": b"",
            }
            for message, metadata in payloads.items():
                with self.subTest(message=message):
                    target = root / "malformed.safetensors"
                    body = multipart_body(
                        [
                            ("package", metadata, "application/json"),
                            (
                                "artifact",
                                b"artifact",
                                "application/octet-stream",
                            ),
                        ]
                    )
                    with self.assertRaisesRegex(
                        KnowledgeTransportError,
                        message,
                    ):
                        await receive_knowledge_transfer(
                            content_type=content_type,
                            chunks=chunks(body),
                            artifact_path=target,
                            metadata_part_name="package",
                            maximum_content_bytes=4096,
                        )
                    self.assertFalse(target.exists())
                    self.assertEqual(list(root.iterdir()), [])

    async def test_content_limit_accepts_exact_size_and_rejects_one_byte_over(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content_type = "multipart/form-data; boundary=test-boundary"
            metadata = b'{"package_hash":"value"}'
            artifact = b"artifact-bytes"
            body = multipart_body(
                [
                    ("package", metadata, "application/json"),
                    ("artifact", artifact, "application/octet-stream"),
                ]
            )
            content_size = len(metadata) + len(artifact)
            accepted = root / "accepted.safetensors"
            received = await receive_knowledge_transfer(
                content_type=content_type,
                chunks=chunks(body),
                artifact_path=accepted,
                metadata_part_name="package",
                maximum_content_bytes=content_size,
            )
            self.assertEqual(received.content_size, content_size)
            self.assertEqual(accepted.read_bytes(), artifact)

            rejected = root / "rejected.safetensors"
            with self.assertRaises(KnowledgeTransportTooLarge):
                await receive_knowledge_transfer(
                    content_type=content_type,
                    chunks=chunks(body),
                    artifact_path=rejected,
                    metadata_part_name="package",
                    maximum_content_bytes=content_size - 1,
                )
            self.assertFalse(rejected.exists())
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["accepted.safetensors"],
            )

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
            self.assertEqual(list(root.iterdir()), [])

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
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
