from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx

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
    ) -> None:
        _, app = stack.client(client_id)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://client",
            headers=stack.client_headers,
        ) as client:
            response = await client.post("/v1/register")

        self.assertEqual(response.status_code, 201, response.text)

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

            create = await stack.coordinator_request(
                "POST",
                "/v1/rounds",
                headers={"X-Admin-Token": stack.admin_token},
                json={
                    "selected_client_ids": ["client-a"],
                    "trusted_client_quorum": 1,
                    "prompt_template": (
                        "Question: {question}\nAnswer:"
                    ),
                    "top_k": 2,
                },
            )

            self.assertEqual(create.status_code, 201, create.text)
            round_id = create.json()["round_id"]
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


if __name__ == "__main__":
    unittest.main()