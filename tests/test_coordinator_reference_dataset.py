from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi import FastAPI

from client.runtime import ClientRuntime, ClientRuntimeError
from shared.protocol import RoundManifest
from shared.reference_dataset import (
    ReferenceSample,
    load_reference_jsonl,
    reference_dataset_identity,
    write_reference_jsonl,
)
from tests.test_round import Stack


def create_dataset_files(
    root: Path,
) -> tuple[Path, Path, list[ReferenceSample]]:
    dataset_dir = root / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    reference = [
        ReferenceSample(
            dataset_id="example-reference",
            dataset_version="v1",
            sample_id="example-ch001-s001-q001",
            chapter="Example Chapter",
            section="Example Section",
            question="What is the first question?",
            gold_answer="The first answer.",
        ),
        ReferenceSample(
            dataset_id="example-reference",
            dataset_version="v1",
            sample_id="example-ch001-s001-q002",
            chapter="Example Chapter",
            section="Example Section",
            question="What is the second question?",
            gold_answer="The second answer.",
        ),
    ]

    validation = [
        ReferenceSample(
            dataset_id="example-reference",
            dataset_version="v1",
            sample_id="example-ch001-s001-q003",
            chapter="Example Chapter",
            section="Example Section",
            question="What is the validation question?",
            gold_answer="The validation answer.",
        )
    ]

    reference_path = dataset_dir / "reference.jsonl"
    validation_path = dataset_dir / "validation.jsonl"

    write_reference_jsonl(reference_path, reference)
    write_reference_jsonl(validation_path, validation)

    return reference_path, validation_path, reference


class CoordinatorReferenceDatasetTests(
    unittest.IsolatedAsyncioTestCase
):
    async def register_client(
        self,
        stack: Stack,
        client_id: str,
    ) -> tuple[ClientRuntime, FastAPI]:
        runtime, app = stack.client(client_id)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://client",
            headers=stack.client_headers,
        ) as client:
            response = await client.post("/v1/register")

        self.assertEqual(response.status_code, 201, response.text)

        return runtime, app

    async def create_real_round(
        self,
        stack: Stack,
        client_id: str = "client-a",
    ) -> dict:
        response = await stack.coordinator_request(
            "POST",
            "/v1/rounds",
            headers={"X-Admin-Token": stack.admin_token},
            json={
                "selected_client_ids": [client_id],
                "trusted_client_quorum": 1,
                "prompt_template": (
                    "Question: {question}\nAnswer:"
                ),
                "top_k": 2,
            },
        )

        self.assertEqual(response.status_code, 201, response.text)

        return response.json()

    async def test_manifest_uses_loaded_reference_dataset(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path, validation_path, reference = (
                create_dataset_files(root)
            )

            stack = Stack(
                directory,
                reference_dataset_path=reference_path,
                validation_dataset_path=validation_path,
            )
            await self.register_client(stack, "client-a")

            response = await stack.coordinator_request(
                "POST",
                "/v1/rounds",
                headers={"X-Admin-Token": stack.admin_token},
                json={
                    "selected_client_ids": ["client-a"],
                    "trusted_client_quorum": 1,
                    "reference_dataset_id": "fabricated",
                    "reference_dataset_hash": "0" * 64,
                    "sample_ids": ["fake"],
                    "prompt_template": (
                        "Question: {question}\nAnswer:"
                    ),
                    "top_k": 2,
                },
            )

            self.assertEqual(
                response.status_code,
                201,
                response.text,
            )

            manifest = response.json()
            identity = reference_dataset_identity(reference)

            self.assertEqual(
                manifest["reference_dataset_id"],
                identity.dataset_id,
            )
            self.assertEqual(
                manifest["reference_dataset_hash"],
                identity.dataset_hash,
            )
            self.assertEqual(
                manifest["sample_ids"],
                [
                    "example-ch001-s001-q001",
                    "example-ch001-s001-q002",
                ],
            )

            round_id = manifest["round_id"]
            snapshot = (
                root
                / "coordinator"
                / "rounds"
                / round_id
                / "datasets"
                / "reference.jsonl"
            )
            validation_snapshot = (
                root
                / "coordinator"
                / "rounds"
                / round_id
                / "datasets"
                / "validation.jsonl"
            )

            self.assertTrue(snapshot.is_file())
            self.assertTrue(validation_snapshot.is_file())
            self.assertEqual(
                [
                    sample.sample_id
                    for sample in load_reference_jsonl(snapshot)
                ],
                manifest["sample_ids"],
            )

    async def test_selected_client_can_download_reference_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path, validation_path, _ = (
                create_dataset_files(root)
            )

            stack = Stack(
                directory,
                reference_dataset_path=reference_path,
                validation_dataset_path=validation_path,
            )

            await self.register_client(stack, "client-a")
            await self.register_client(stack, "client-b")

            manifest = await self.create_real_round(stack)
            round_id = manifest["round_id"]
            endpoint = (
                f"/v1/rounds/{round_id}/reference-dataset"
            )

            downloaded = await stack.coordinator_request(
                "GET",
                endpoint,
                headers={
                    "X-Client-ID": "client-a",
                    "X-Registration-Token": (
                        stack.registration_token
                    ),
                },
            )

            self.assertEqual(
                downloaded.status_code,
                200,
                downloaded.text,
            )
            self.assertTrue(
                downloaded.headers["content-type"].startswith(
                    "application/x-ndjson"
                )
            )

            rows = [
                json.loads(line)
                for line in downloaded.text.splitlines()
            ]
            self.assertEqual(
                [row["sample_id"] for row in rows],
                [
                    "example-ch001-s001-q001",
                    "example-ch001-s001-q002",
                ],
            )

            unselected = await stack.coordinator_request(
                "GET",
                endpoint,
                headers={
                    "X-Client-ID": "client-b",
                    "X-Registration-Token": (
                        stack.registration_token
                    ),
                },
            )
            self.assertEqual(unselected.status_code, 403)

            missing_token = await stack.coordinator_request(
                "GET",
                endpoint,
                headers={"X-Client-ID": "client-a"},
            )
            self.assertEqual(missing_token.status_code, 401)

    async def test_client_participation_caches_verified_reference_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path, validation_path, _ = (
                create_dataset_files(root)
            )

            stack = Stack(
                directory,
                reference_dataset_path=reference_path,
                validation_dataset_path=validation_path,
            )
            _, app = await self.register_client(
                stack,
                "client-a",
            )
            manifest = await self.create_real_round(stack)
            round_id = manifest["round_id"]

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client-a",
                headers=stack.client_headers,
            ) as client:
                participate = await client.post(
                    "/v1/participate"
                )

            self.assertEqual(
                participate.status_code,
                201,
                participate.text,
            )

            client_cache = (
                root
                / "client-a"
                / "reference_datasets"
                / round_id
                / "reference.jsonl"
            )
            client_identity = (
                root
                / "client-a"
                / "reference_datasets"
                / round_id
                / "identity.json"
            )

            self.assertTrue(client_cache.is_file())
            self.assertTrue(client_identity.is_file())

            cached = load_reference_jsonl(client_cache)

            self.assertEqual(
                [
                    sample.sample_id
                    for sample in cached
                ],
                manifest["sample_ids"],
            )
            self.assertEqual(
                reference_dataset_identity(
                    cached
                ).dataset_hash,
                manifest["reference_dataset_hash"],
            )

            identity_record = json.loads(
                client_identity.read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                identity_record["round_id"],
                round_id,
            )
            self.assertEqual(
                identity_record["manifest_hash"],
                manifest["manifest_hash"],
            )

    async def test_client_rejects_tampered_reference_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path, validation_path, _ = (
                create_dataset_files(root)
            )

            stack = Stack(
                directory,
                reference_dataset_path=reference_path,
                validation_dataset_path=validation_path,
            )
            runtime, _ = await self.register_client(
                stack,
                "client-a",
            )
            manifest_payload = await self.create_real_round(
                stack
            )
            manifest = RoundManifest.model_validate(
                manifest_payload
            )

            original = reference_path.read_bytes()
            tampered = original.replace(
                b"The first answer.",
                b"A changed first answer.",
            )

            self.assertNotEqual(tampered, original)

            with self.assertRaises(ClientRuntimeError):
                runtime.cache_reference_dataset(
                    manifest=manifest,
                    content=tampered,
                )

            cache_root = (
                root
                / "client-a"
                / "reference_datasets"
                / manifest.round_id
            )

            self.assertFalse(
                (cache_root / "reference.jsonl").exists()
            )
            self.assertFalse(
                (cache_root / "identity.json").exists()
            )

    async def test_host_caches_verified_reference_and_validation_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path, validation_path, _ = (
                create_dataset_files(root)
            )

            stack = Stack(
                directory,
                reference_dataset_path=reference_path,
                validation_dataset_path=validation_path,
            )
            _, app = await self.register_client(
                stack,
                "client-a",
            )
            manifest = await self.create_real_round(stack)
            round_id = manifest["round_id"]

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client-a",
                headers=stack.client_headers,
            ) as client:
                participate = await client.post(
                    "/v1/participate"
                )

            self.assertEqual(
                participate.status_code,
                201,
                participate.text,
            )
            self.assertEqual(
                participate.json()["state"],
                "COMPLETED",
            )

            host_dataset_root = (
                root
                / "host"
                / "rounds"
                / round_id
                / "datasets"
            )
            host_reference_path = (
                host_dataset_root / "reference.jsonl"
            )
            host_validation_path = (
                host_dataset_root / "validation.jsonl"
            )
            host_identity_path = (
                host_dataset_root / "identity.json"
            )

            self.assertTrue(host_reference_path.is_file())
            self.assertTrue(host_validation_path.is_file())
            self.assertTrue(host_identity_path.is_file())

            host_reference = load_reference_jsonl(
                host_reference_path
            )
            host_validation = load_reference_jsonl(
                host_validation_path
            )

            self.assertEqual(
                reference_dataset_identity(
                    host_reference
                ).dataset_hash,
                manifest["reference_dataset_hash"],
            )
            self.assertEqual(
                [
                    sample.sample_id
                    for sample in host_reference
                ],
                manifest["sample_ids"],
            )
            self.assertEqual(
                [
                    sample.sample_id
                    for sample in host_validation
                ],
                ["example-ch001-s001-q003"],
            )
            self.assertFalse(
                {
                    sample.sample_id
                    for sample in host_reference
                }
                & {
                    sample.sample_id
                    for sample in host_validation
                }
            )

            identity_record = json.loads(
                host_identity_path.read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                identity_record["round_id"],
                round_id,
            )
            self.assertEqual(
                identity_record["manifest_hash"],
                manifest["manifest_hash"],
            )
            self.assertEqual(
                identity_record["reference"]["dataset_hash"],
                manifest["reference_dataset_hash"],
            )
            self.assertEqual(
                identity_record["validation"]["dataset_hash"],
                reference_dataset_identity(
                    host_validation
                ).dataset_hash,
            )


if __name__ == "__main__":
    unittest.main()
