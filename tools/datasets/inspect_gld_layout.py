from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pymupdf


DEFAULT_PDF_PATH = Path("data/private/greek_law_digest.pdf")
DEFAULT_OUTPUT_PATH = Path("data/derived/gld_layout_inspection.json")

DEFAULT_ANCHORS = (
    "PROCEDURE BEFORE CIVIL COURTS",
    "PROCEDURE BEFORE ADMINISTRATIVE COURTS",
    "SOCIETE ANONYME - COMPANY LIMITED BY SHARES",
    "NEGOTIABLE INSTRUMENTS",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_search_text(value: str) -> str:
    return " ".join(value.upper().split())


def rounded_bbox(value: Any) -> list[float]:
    return [round(float(number), 3) for number in value]


def extract_page_lines(page: pymupdf.Page) -> list[dict[str, Any]]:
    payload = page.get_text("dict", sort=True)
    lines: list[dict[str, Any]] = []

    for block in payload.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans)

            if not text.strip():
                continue

            lines.append(
                {
                    "text": text,
                    "bbox": rounded_bbox(line["bbox"]),
                    "spans": [
                        {
                            "text": span.get("text", ""),
                            "font": span.get("font", ""),
                            "size": round(float(span.get("size", 0.0)), 3),
                            "flags": int(span.get("flags", 0)),
                            "color": int(span.get("color", 0)),
                            "bbox": rounded_bbox(span["bbox"]),
                        }
                        for span in spans
                    ],
                }
            )

    return lines


def inspect_pdf(
    pdf_path: Path,
    anchors: list[str],
    requested_pages: list[int],
) -> dict[str, Any]:
    normalized_anchors = {
        anchor: normalize_search_text(anchor)
        for anchor in anchors
    }

    with pymupdf.open(pdf_path) as document:
        matches: dict[str, list[int]] = {
            anchor: []
            for anchor in anchors
        }
        page_texts: dict[int, str] = {}

        for page_index, page in enumerate(document):
            pdf_page = page_index + 1
            text = page.get_text("text", sort=True)
            normalized_text = normalize_search_text(text)
            page_texts[pdf_page] = text

            for anchor, normalized_anchor in normalized_anchors.items():
                if normalized_anchor in normalized_text:
                    matches[anchor].append(pdf_page)

        pages_to_inspect = set(requested_pages)

        for matching_pages in matches.values():
            pages_to_inspect.update(matching_pages)

        invalid_pages = [
            page
            for page in pages_to_inspect
            if page < 1 or page > document.page_count
        ]

        if invalid_pages:
            raise ValueError(
                f"PDF pages outside valid range: {sorted(invalid_pages)}"
            )

        inspected_pages = []

        for pdf_page in sorted(pages_to_inspect):
            page = document[pdf_page - 1]

            inspected_pages.append(
                {
                    "pdf_page": pdf_page,
                    "width": round(float(page.rect.width), 3),
                    "height": round(float(page.rect.height), 3),
                    "plain_text": page_texts[pdf_page],
                    "lines": extract_page_lines(page),
                }
            )

        return {
            "source": {
                "filename": pdf_path.name,
                "sha256": sha256_file(pdf_path),
                "page_count": document.page_count,
            },
            "extractor": {
                "name": "PyMuPDF",
                "version": getattr(
                    pymupdf,
                    "VersionBind",
                    "unknown",
                ),
                "text_sort": True,
            },
            "anchors": anchors,
            "matches": [
                {
                    "anchor": anchor,
                    "pdf_pages": matches[anchor],
                }
                for anchor in anchors
            ],
            "pages": inspected_pages,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Greek Law Digest PDF layout metadata."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--anchor",
        action="append",
        dest="anchors",
    )
    parser.add_argument(
        "--page",
        action="append",
        type=int,
        dest="pages",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path: Path = args.pdf
    output_path: Path = args.output
    anchors: list[str] = args.anchors or list(DEFAULT_ANCHORS)
    pages: list[int] = args.pages or []

    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    report = inspect_pdf(
        pdf_path=pdf_path,
        anchors=anchors,
        requested_pages=pages,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Source SHA-256: {report['source']['sha256']}")
    print(f"PDF pages: {report['source']['page_count']}")

    for match in report["matches"]:
        print(
            f"{match['anchor']}: "
            f"{match['pdf_pages'] or 'not found'}"
        )

    print(f"Inspection report: {output_path}")


if __name__ == "__main__":
    main()