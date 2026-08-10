from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.crypto import Ed25519Identity, canonical_json_bytes, sha256_hex
from shared.fedmkt_runtime import deterministic_knowledge_samples
from shared.knowledge_artifact import serialize_knowledge_artifact
from shared.knowledge_transport import (
    KnowledgeTransportTooLarge,
    receive_knowledge_transfer,
)
from shared.protocol import (
    KnowledgePackage,
    KnowledgeSample,
    LoraProfile,
    ModelProfile,
    RoundCreateRequest,
    RoundManifest,
    utc_now,
    utc_text,
)
from shared.reference_dataset import (
    load_reference_jsonl,
    reference_dataset_identity,
)

DEFAULT_REFERENCE_PATH = Path("data/derived/gld2012/reference.jsonl")
EXPECTED_SAMPLE_COUNT = 565
EXPECTED_DATASET_HASH = (
    "5d855a429d43b70eb146aeb11cda1f675c05d6465bea0792796fdcd8d6ceb231"
)


def _measurement_profile(role: str) -> ModelProfile:
    return ModelProfile(
        profile_id=f"{role}-mock-v1",
        role=role,
        model_id=f"legalfedllm/mock-{role}",
        model_revision="mock-v1",
        tokenizer_id="legalfedllm/mock-tokenizer",
        tokenizer_revision="mock-v1",
        tokenizer_class="MockTokenizer",
        training_backend="mock",
        serving_backend="mock",
        prompt_template_hash=sha256_hex(b"legalfedllm-default-prompt"),
        lora=LoraProfile(
            rank=8,
            alpha=16,
            target_modules=("q_proj", "v_proj"),
        ),
    )


def _multipart_body(
    package: KnowledgePackage,
    artifact: bytes,
) -> tuple[str, bytes]:
    request = httpx.Request(
        "POST",
        "http://measurement.invalid/v1/knowledge",
        files=[
            (
                "package",
                (
                    "package.json",
                    canonical_json_bytes(package.model_dump(mode="json")),
                    "application/json",
                ),
            ),
            (
                "artifact",
                (
                    "knowledge.safetensors",
                    artifact,
                    "application/octet-stream",
                ),
            ),
        ],
    )
    return request.headers["Content-Type"], request.read()


async def _chunks(value: bytes) -> AsyncIterator[bytes]:
    for start in range(0, len(value), 4096):
        yield value[start : start + 4096]


async def _transport_accepts(
    *,
    content_type: str,
    body: bytes,
    maximum_content_bytes: int,
) -> bool:
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "knowledge.safetensors"
        try:
            await receive_knowledge_transfer(
                content_type=content_type,
                chunks=_chunks(body),
                artifact_path=target,
                metadata_part_name="package",
                maximum_content_bytes=maximum_content_bytes,
            )
        except KnowledgeTransportTooLarge:
            if target.exists() or list(target.parent.glob("*.part")):
                raise RuntimeError("rejected transport left temporary files")
            return False
        return target.is_file()


def _embedded_v1_bytes(
    *,
    package: KnowledgePackage,
    samples: list[KnowledgeSample],
    identity: Ed25519Identity,
) -> bytes:
    payload = package.model_dump(
        mode="json",
        exclude={"artifact", "package_hash", "signature"},
    )
    payload["package_schema_version"] = "1.0"
    payload["samples"] = [sample.model_dump(mode="json") for sample in samples]
    artifact_sha256 = sha256_hex(payload)
    signed_payload = {**payload, "artifact_sha256": artifact_sha256}
    equivalent = {
        **signed_payload,
        "signature": identity.sign_json(signed_payload),
    }
    return canonical_json_bytes(equivalent)


async def _measure_package(
    *,
    package: KnowledgePackage,
    samples: list[KnowledgeSample],
    artifact: bytes,
    identity: Ed25519Identity,
    configured_maximum_bytes: int,
) -> dict[str, Any]:
    metadata = canonical_json_bytes(package.model_dump(mode="json"))
    embedded = _embedded_v1_bytes(
        package=package,
        samples=samples,
        identity=identity,
    )
    content_type, multipart = _multipart_body(package, artifact)
    logical_size = len(metadata) + len(artifact)
    reduction = len(embedded) - logical_size

    configured_accepts = await _transport_accepts(
        content_type=content_type,
        body=multipart,
        maximum_content_bytes=configured_maximum_bytes,
    )
    exact_accepts = await _transport_accepts(
        content_type=content_type,
        body=multipart,
        maximum_content_bytes=logical_size,
    )
    below_accepts = await _transport_accepts(
        content_type=content_type,
        body=multipart,
        maximum_content_bytes=logical_size - 1,
    )

    return {
        "artifact_sha256": package.artifact.sha256,
        "sample_count": package.artifact.sample_count,
        "total_token_count": package.artifact.total_token_count,
        "top_k": package.top_k,
        "sizes": {
            "package_json_bytes": len(metadata),
            "safetensors_bytes": len(artifact),
            "logical_package_bytes": logical_size,
            "multipart_body_bytes": len(multipart),
            "multipart_framing_bytes": len(multipart) - logical_size,
            "equivalent_embedded_json_bytes": len(embedded),
            "reduction_bytes": reduction,
            "reduction_percent": round(
                reduction * 100 / len(embedded),
                4,
            ),
        },
        "limit_probe": {
            "configured_maximum_bytes": configured_maximum_bytes,
            "configured_limit_result": (
                "accepted" if configured_accepts else "rejected"
            ),
            "exact_content_limit_bytes": logical_size,
            "exact_content_limit_result": (
                "accepted" if exact_accepts else "rejected"
            ),
            "one_byte_below_limit_bytes": logical_size - 1,
            "one_byte_below_limit_result": (
                "accepted" if below_accepts else "rejected"
            ),
        },
    }


async def measure(
    *,
    reference_path: str | Path,
    expected_sample_count: int,
    expected_dataset_hash: str,
    top_k: int,
    maximum_sequence_length: int,
    maximum_knowledge_package_bytes: int,
) -> dict[str, Any]:
    reference_samples = load_reference_jsonl(reference_path)
    dataset = reference_dataset_identity(reference_samples)
    if dataset.sample_count != expected_sample_count:
        raise ValueError(
            f"expected {expected_sample_count} D^P samples, "
            f"found {dataset.sample_count}"
        )
    if dataset.dataset_hash != expected_dataset_hash:
        raise ValueError(
            "D^P semantic hash does not match the accepted frozen dataset"
        )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        coordinator_identity = Ed25519Identity.load_or_create(
            root / "coordinator.pem"
        )
        client_identity = Ed25519Identity.load_or_create(
            root / "client.pem"
        )
        host_identity = Ed25519Identity.load_or_create(root / "host.pem")
        request = RoundCreateRequest(
            selected_client_ids=["measurement-client"],
            trusted_client_quorum=1,
            reference_dataset_id=dataset.dataset_id,
            reference_dataset_hash=dataset.dataset_hash,
            sample_ids=[sample.sample_id for sample in reference_samples],
            prompt_template="Question: {question}\nAnswer: {answer}",
            maximum_sequence_length=maximum_sequence_length,
            top_k=top_k,
            maximum_knowledge_package_bytes=maximum_knowledge_package_bytes,
        )
        manifest = RoundManifest.create_signed(
            identity=coordinator_identity,
            round_id="step-2-package-measurement",
            coordinator_id="measurement-coordinator",
            current_host_adapter_version=0,
            host_model_profile=_measurement_profile("host"),
            request=request,
            submission_deadline=utc_text(utc_now() + timedelta(hours=1)),
        )

        packages: dict[str, dict[str, Any]] = {}
        for role, participant_id, profile, identity in (
            (
                "client",
                "measurement-client",
                _measurement_profile("client"),
                client_identity,
            ),
            (
                "host",
                "measurement-host",
                _measurement_profile("host"),
                host_identity,
            ),
        ):
            samples = deterministic_knowledge_samples(
                manifest=manifest,
                participant_id=participant_id,
                role=role,
                adapter_version=0,
            )
            artifact, descriptor = serialize_knowledge_artifact(samples)
            package = KnowledgePackage.create_signed(
                identity=identity,
                round_id=manifest.round_id,
                manifest_hash=manifest.manifest_hash,
                sender_id=participant_id,
                sender_role=role,
                model_profile=profile,
                adapter_version=0,
                alignment_profile_id="mock_identity:1",
                reference_dataset_id=dataset.dataset_id,
                reference_dataset_hash=dataset.dataset_hash,
                top_k=top_k,
                sample_ids=manifest.sample_ids,
                artifact=descriptor,
                nonce=f"step-2-{role}-measurement-nonce",
                created_at="2026-08-10T00:00:00Z",
            )
            packages[role] = await _measure_package(
                package=package,
                samples=samples,
                artifact=artifact,
                identity=identity,
                configured_maximum_bytes=maximum_knowledge_package_bytes,
            )

    return {
        "measurement_schema_version": "1.0",
        "representation": (
            "signed package metadata plus hash-bound safetensors artifact"
        ),
        "comparison": (
            "pre-Step-2 schema 1.0 package with the same mock samples "
            "embedded in canonical JSON"
        ),
        "size_definitions": {
            "logical_package_bytes": (
                "canonical package JSON plus safetensors bytes"
            ),
            "multipart_body_bytes": (
                "multipart body including framing and excluding HTTP headers"
            ),
            "manifest_limit": (
                "applies to logical package bytes; multipart framing is "
                "bounded separately"
            ),
        },
        "dataset": dataset.model_dump(mode="json"),
        "mock_generation": {
            "maximum_sequence_length": maximum_sequence_length,
            "effective_sequence_length": min(12, maximum_sequence_length),
            "top_k": top_k,
        },
        "packages": packages,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Step 2 Client and Host Knowledge Package representations "
            "using the frozen D^P dataset."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expected-sample-count",
        type=int,
        default=EXPECTED_SAMPLE_COUNT,
    )
    parser.add_argument(
        "--expected-dataset-hash",
        default=EXPECTED_DATASET_HASH,
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--maximum-sequence-length", type=int, default=512)
    parser.add_argument(
        "--maximum-knowledge-package-bytes",
        type=int,
        default=25 * 1024 * 1024,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.reference.is_file():
        print(f"D^P JSONL not found: {args.reference}", file=sys.stderr)
        return 2
    try:
        result = asyncio.run(
            measure(
                reference_path=args.reference,
                expected_sample_count=args.expected_sample_count,
                expected_dataset_hash=args.expected_dataset_hash,
                top_k=args.top_k,
                maximum_sequence_length=args.maximum_sequence_length,
                maximum_knowledge_package_bytes=(
                    args.maximum_knowledge_package_bytes
                ),
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rendered = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
