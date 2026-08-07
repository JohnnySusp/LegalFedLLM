from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.reference_dataset import (
    ReferenceSample,
    ReferenceSource,
    reference_dataset_identity,
    split_reference_samples,
    write_pretty_reference_json,
    write_reference_jsonl,
)

DEFAULT_PDF_PATH = Path("data/private/greek_law_digest.pdf")
DEFAULT_OUTPUT_DIR = Path("data/derived/gld2012")
DEFAULT_DATASET_ID = "gld-internal"
DEFAULT_DATASET_VERSION = "2012-v1"
DEFAULT_DOCUMENT_ID = "gld-2012"
DEFAULT_LAST_INCLUDED_PRINTED_PAGE = 305
EXPECTED_SOURCE_SHA256 = "9673ee7c86b3d582e2c08e1cdd2b84f144981f31a1fe50d4216e82c5b350b77d"
EXPECTED_SOURCE_PAGE_COUNT = 713

BLUE_TEXT_COLOR = 44783

QUESTION_START_RE = re.compile(
    r"^(?:\d+[.)]\s*)?(?:"
    r"what|which|who|whom|whose|when|where|why|how|"
    r"is|are|was|were|can|could|does|do|did|will|would|"
    r"should|may|must|please|under|in\s+which|to\s+what"
    r")\b",
    re.IGNORECASE,
)

PROFILE_MARKERS = (
    "TEL",
    "FAX",
    "E-MAIL",
    "EMAIL",
    "URL",
    "LANGUAGES",
    "CONTACT",
    "AREAS OF PRACTICE",
    "NUMBER OF LAWYERS",
)

EXTRACTION_HYPHENS = ("\ufffe", "\ufffd", "\u00ad")


@dataclass(frozen=True)
class SectionSpec:
    chapter_number: int
    section_number: int
    chapter: str
    section: str
    printed_start_page: int
    firm: str
    anchor: str
    qa: bool = True


@dataclass(frozen=True)
class LineRecord:
    text: str
    pdf_page: int
    printed_page: int
    block_index: int = 0
    line_index: int = 0
    x0: float = 42.5
    x1: float = 400.0
    y0: float = 100.0
    y1: float = 110.0
    page_height: float = 842.0
    bold_ratio: float = 0.0
    font_size: float = 10.0
    color: int = 0
    font_name: str = ""


@dataclass(frozen=True)
class QuestionGroup:
    start_index: int
    end_index: int
    text: str


@dataclass(frozen=True)
class QuestionAnswerPair:
    source_ordinals: tuple[int, ...]
    question: str
    answer: str
    page_start: int
    page_end: int


@dataclass(frozen=True)
class FollowUpDecision:
    merge: bool
    review: bool
    reason: str | None = None


@dataclass
class SectionParseResult:
    samples: list[ReferenceSample]
    warnings: list[str]
    blocking_issues: list[str]
    groups: list[dict[str, Any]]
    review_candidates: list[dict[str, Any]]
    embedded_question_lines: list[dict[str, Any]]


SECTION_SPECS: tuple[SectionSpec, ...] = (
    SectionSpec(1, 1, "JUDICIAL SYSTEM", "PROCEDURE BEFORE CIVIL COURTS", 34, "Kelemenis & Co.", "PROCEDURE BEFORE CIVIL COURTS"),
    SectionSpec(1, 2, "JUDICIAL SYSTEM", "PROCEDURE BEFORE ADMINISTRATIVE COURTS", 40, "Papageorgiou Alexandra & Partners", "PROCEDURE BEFORE ADMINISTRATIVE COURTS"),
    SectionSpec(1, 3, "JUDICIAL SYSTEM", "PROCEDURE BEFORE CRIMINAL COURTS", 46, "Anagnostopoulos Criminal Law & Litigation", "PROCEDURE BEFORE CRIMINAL COURTS"),
    SectionSpec(1, 4, "JUDICIAL SYSTEM", "ARBITRATION UNDER GREEK CODE OF CIVIL PROCEDURE", 52, "Kouvela-Piquet & Associates", "ARBITRATION UNDER GREEK CODE OF CIVIL PROCEDURE"),
    SectionSpec(1, 5, "JUDICIAL SYSTEM", "ALTERNATIVE DISPUTE RESOLUTION - MEDIATION", 58, "Dimitra K. Triantafyllou - Delta to the Epsilon", "ALTERNATIVE DISPUTE RESOLUTION - MEDIATION"),
    SectionSpec(1, 6, "JUDICIAL SYSTEM", "ENFORCEMENT OF: FOREIGN JUDGMENTS AND FOREIGN ARBITRAL AWARDS IN GREECE", 64, "Meidanis Pagoulatos & Associates Law Office", "ENFORCEMENT OF: FOREIGN JUDGMENTS AND FOREIGN ARBITRAL AWARDS IN GREECE"),
    SectionSpec(1, 7, "JUDICIAL SYSTEM", "PROCEDURES BEFORE EUROPEAN COURTS", 70, "Christianos & Partners Law Firm", "PROCEDURES BEFORE EUROPEAN COURTS"),

    SectionSpec(2, 1, "BASIC ASPECTS OF CIVIL LAW", "GENERAL PRINCIPLES OF CONTRACT LAW", 76, "A.S. Papadimitriou & Partners Law Firm", "GENERAL PRINCIPLES OF CONTRACT LAW"),
    SectionSpec(2, 2, "BASIC ASPECTS OF CIVIL LAW", "SALE OF GOODS", 82, "Nikas, Theisen & Associates LLP", "SALE OF GOODS"),
    SectionSpec(2, 3, "BASIC ASPECTS OF CIVIL LAW", "UNJUST ENRICHMENT", 88, "Roussos & Partners", "UNJUST ENRICHMENT"),
    SectionSpec(2, 4, "BASIC ASPECTS OF CIVIL LAW", "ESTATES - WILLS - HEIRS / Civil Law provisions - Tax Considerations", 92, "Vounatsos Attorneys", "ESTATES - WILLS - HEIRS"),
    SectionSpec(2, 5, "BASIC ASPECTS OF CIVIL LAW", "FAMILY LAW", 98, "Law Office George Papacharalampous - Ifigeneia Spanidi & Partners", "FAMILY LAW"),
    SectionSpec(2, 6, "BASIC ASPECTS OF CIVIL LAW", "TORT, PERSONAL INJURY & COMPENSATION", 104, "Pavlakis - Moschos & Associates", "TORT, PERSONAL INJURY & COMPENSATION"),

    SectionSpec(3, 1, "BUSINESS ENTITIES", "SOCIETE ANONYME - COMPANY LIMITED BY SHARES / General Provisions - Administration", 112, "Kelemenis & Co.", "General Provisions - Administration"),
    SectionSpec(3, 2, "BUSINESS ENTITIES", "SOCIETE ANONYME - COMPANY LIMITED BY SHARES / Accounting / Audit for S.A. (for non listed)", 118, "Nomos Law Firm", "Accounting / Audit for SA"),
    SectionSpec(3, 3, "BUSINESS ENTITIES", "SOCIETE ANONYME - COMPANY LIMITED BY SHARES / Minority Shareholders Rights - Shareholders Agreements", 124, "Dryllerakis & Associates", "Minority Shareholders Rights - Shareholders Agreements"),
    SectionSpec(3, 4, "BUSINESS ENTITIES", "SOCIETE ANONYME - COMPANY LIMITED BY SHARES / Tax Issues", 130, "Dryllerakis & Associates", "Tax Issues"),
    SectionSpec(3, 5, "BUSINESS ENTITIES", "SOCIETE ANONYME - COMPANY LIMITED BY SHARES / Corporations listed on the ATHEX", 136, "Lambadarios Law Firm", "Corporations listed on the ATHEX"),
    SectionSpec(3, 6, "BUSINESS ENTITIES", "LIMITED LIABILITY COMPANY, L.T.D. (E.P.E.) / Formation - Capital requirements - Administration - Distribution of profits - Liquidation", 142, "Spiridonos Law Firm", "LIMITED LIABILITY COMPANY, L.T.D. (E.P.E.)"),
    SectionSpec(3, 7, "BUSINESS ENTITIES", "LIMITED LIABILITY COMPANY L.T.D. (E.P.E.) / Accounting books and records - Audit requirements - Tax issues", 148, "Iason Skouzos + Partners Law Firm", "Accounting books and records - Audit requirements - Tax issues", False),
    SectionSpec(3, 8, "BUSINESS ENTITIES", "PARTNERSHIPS / General Partnership (OE) - Limited Partnership (EE) - Joint Venture", 151, "Athanasios Kikis & Partners Law Office", "PARTNERSHIPS"),
    SectionSpec(3, 9, "BUSINESS ENTITIES", "OTHER BUSINESS STRUCTURES", 157, "Kouvela-Piquet & Associates", "OTHER BUSINESS STRUCTURES"),
    SectionSpec(3, 10, "BUSINESS ENTITIES", "INVESTING THROUGH A LOW TAX JURISDICTION STRUCTURE", 163, "Vardikos & Vardikos Attorneys & Counsellors at Law, Tax Consultants", "INVESTING THROUGH A LOW TAX JURISDICTION STRUCTURE", False),
    SectionSpec(3, 11, "BUSINESS ENTITIES", "MUTUAL FUNDS - PORTFOLIO INVESTMENT COMPANIES - VENTURE CAPITAL", 167, "A.S. Papadimitriou & Partners Law Firm", "MUTUAL FUNDS - PORTFOLIO INVESTMENT COMPANIES - VENTURE CAPITAL"),
    SectionSpec(3, 12, "BUSINESS ENTITIES", "BANKING ENTERPRISES", 173, "Papapolitis & Papapolitis", "BANKING ENTERPRISES"),
    SectionSpec(3, 13, "BUSINESS ENTITIES", "HOW TO SET UP AN SA IN GREECE THROUGH THE NEW ONE STOP SHOP PROCEDURE", 179, "Vounatsos Attorneys", "HOW TO SET UP AN SA IN GREECE THROUGH THE NEW ONE STOP SHOP PROCEDURE"),

    SectionSpec(4, 1, "BANKING SYSTEM - FINANCE - INVESTMENT", "BANKING SYSTEM", 186, "Lambadarios Law Firm", "BANKING SYSTEM"),
    SectionSpec(4, 2, "BANKING SYSTEM - FINANCE - INVESTMENT", "INVESTMENT INCENTIVES LAW", 192, "Avgerinos & Partners Law Firm", "INVESTMENT INCENTIVES LAW"),
    SectionSpec(4, 3, "BANKING SYSTEM - FINANCE - INVESTMENT", "FAST TRACK LAW", 198, "Avgerinos & Partners Law Firm", "FAST TRACK LAW"),
    SectionSpec(4, 4, "BANKING SYSTEM - FINANCE - INVESTMENT", "PUBLIC PROCUREMENT & PROJECTS", 204, "Vg Lawyers Vrettos - Ganiatsos & Associates", "PUBLIC PROCUREMENT & PROJECTS"),
    SectionSpec(4, 5, "BANKING SYSTEM - FINANCE - INVESTMENT", "PRIVATE PUBLIC PARTNERSHIPS (LAW 3389/2005)", 210, "Lambadarios Law Firm", "PRIVATE PUBLIC PARTNERSHIPS"),
    SectionSpec(4, 6, "BANKING SYSTEM - FINANCE - INVESTMENT", "PRIVATIZATIONS", 214, "Lykourezos Law Offices", "PRIVATIZATIONS"),
    SectionSpec(4, 7, "BANKING SYSTEM - FINANCE - INVESTMENT", "SECURITIZATION LAW (LAW 3156/2003)", 218, "Sardelas Liarikos & Associates Law Firm", "SECURITIZATION LAW"),
    SectionSpec(4, 8, "BANKING SYSTEM - FINANCE - INVESTMENT", "PUBLIC CONTRACTS AND COMPETITION LAW", 224, "M. & P. Bernitsas Law Offices", "PUBLIC CONTRACTS AND COMPETITION LAW"),
    SectionSpec(4, 9, "BANKING SYSTEM - FINANCE - INVESTMENT", "LEGAL PROTECTION OF THE PARTIES PARTICIPATING IN THE AWARD PROCEDURE OF PUBLIC CONTRACTS", 230, "Varotsos & Varotsos Law Offices", "LEGAL PROTECTION OF THE PARTIES PARTICIPATING IN THE AWARD PROCEDURE OF PUBLIC CONTRACTS"),
    SectionSpec(4, 10, "BANKING SYSTEM - FINANCE - INVESTMENT", "FINANCING FOR THE IMPLEMENTATION OF INFRASTRUCTURE PROJECTS", 236, "Karatzas & Partners Law Firm", "FINANCING FOR THE IMPLEMENTATION OF INFRASTRUCTURE PROJECTS"),

    SectionSpec(5, 1, "MERGERS & ACQUISITIONS", "MERGERS - TRANSFORMATIONS OF COMPANIES", 244, "Karamanolis & Associates Law Firm", "MERGERS - TRANSFORMATIONS OF COMPANIES"),
    SectionSpec(5, 2, "MERGERS & ACQUISITIONS", "PRE-MERGER NOTIFICATION", 250, "A.S. Papadimitriou & Partners Law Firm", "PRE-MERGER NOTIFICATION"),
    SectionSpec(5, 3, "MERGERS & ACQUISITIONS", "SPIN OFFS - TRANSFER OF BUSINESS SECTORS OR AGGREGATES OF ASSETS & LIABILITIES", 256, "Kelemenis & Co.", "SPIN OFFS - TRANSFER OF BUSINESS SECTORS OR AGGREGATES OF ASSETS & LIABILITIES"),
    SectionSpec(5, 4, "MERGERS & ACQUISITIONS", "SHARE TRANSFER DEALS", 262, "Kelemenis & Co.", "SHARE TRANSFER DEALS"),
    SectionSpec(5, 5, "MERGERS & ACQUISITIONS", "MANDATORY AND VOLUNTARY TAKEOVER BIDS", 268, "Karatzas & Partners Law Firm", "MANDATORY AND VOLUNTARY TAKEOVER BIDS"),

    SectionSpec(6, 1, "FINANCIAL CONTRACTS", "AGENCY & DISTRIBUTION AGREEMENTS / Legal framework - Tax considerations", 276, "Bahas, Gramatidis & Partners", "AGENCY & DISTRIBUTION AGREEMENTS"),
    SectionSpec(6, 2, "FINANCIAL CONTRACTS", "FRANCHISING / Legal Framework - Tax Considerations", 282, "Bletas & Costakis Law Firm", "FRANCHISING"),
    SectionSpec(6, 3, "FINANCIAL CONTRACTS", "LEASING / Legal Framework - Tax Considerations", 287, "A.S. Papadimitriou & Partners Law Firm", "LEASING"),
    SectionSpec(6, 4, "FINANCIAL CONTRACTS", "FACTORING - FORFAITING (FORFEITING)", 293, "Papageorgiou Alexandra & Partners", "FACTORING - FORFAITING"),

    SectionSpec(7, 1, "FINANCIAL TOOLS", "NEGOTIABLE INSTRUMENTS", 300, "Papageorgiou Alexandra & Partners", "NEGOTIABLE INSTRUMENTS"),

    # Sentinel only: the first section after the selected corpus boundary.
    SectionSpec(7, 2, "FINANCIAL TOOLS", "COVERED BONDS", 306, "Karatzas & Partners Law Firm", "COVERED BONDS"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_extracted_text(value: str) -> str:
    text = value
    for marker in EXTRACTION_HYPHENS:
        text = text.replace(marker, "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return " ".join(text.split())


def normalize_match_text(value: str) -> str:
    text = normalize_extracted_text(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return " ".join(text.split())


def is_bold(line: LineRecord) -> bool:
    return line.bold_ratio >= 0.55


def is_primary_question_style(line: LineRecord) -> bool:
    font = line.font_name.lower()
    return (
        line.color == BLUE_TEXT_COLOR
        and ("semibold" in font or "bold" in font or is_bold(line))
        and 9.0 <= line.font_size <= 11.0
        and line.x0 <= 50.0
    )


def is_secondary_question_style(line: LineRecord) -> bool:
    font = line.font_name.lower()
    return (
        line.color == 0
        and ("bold" in font or is_bold(line))
        and 9.0 <= line.font_size <= 11.0
        and 58.0 <= line.x0 <= 72.0
    )


def question_style(line: LineRecord) -> str | None:
    if is_primary_question_style(line):
        return "primary"
    if is_secondary_question_style(line):
        return "secondary"
    return None


def is_upper_heading(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and all(char.isupper() for char in letters)


def is_structural_heading(line: LineRecord) -> bool:
    text = normalize_extracted_text(line.text)
    if text.endswith("?"):
        return False
    if is_primary_question_style(line):
        return True
    if is_secondary_question_style(line):
        return True
    return (
        is_bold(line)
        and len(text) <= 180
        and is_upper_heading(text)
    )


def is_running_matter(line: LineRecord, firm: str) -> bool:
    text = normalize_extracted_text(line.text)
    normalized = normalize_match_text(text)
    near_top = line.y0 <= line.page_height * 0.10
    near_bottom = line.y1 >= line.page_height * 0.90

    if near_bottom and "GREEK LAW DIGEST" in normalized:
        return True
    if (near_top or near_bottom) and text.isdigit():
        return True

    page = str(line.printed_page)
    if near_top and page in text and normalize_match_text(firm) in normalized:
        return True

    return False


def is_embedded_question_line(line: LineRecord) -> bool:
    text = normalize_extracted_text(line.text)
    return text.endswith("?") and question_style(line) is None


FOLLOW_UP_AUTO_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("but-what", re.compile(r"^but\s+what\b", re.IGNORECASE)),
    ("if-so", re.compile(r"^if\s+so\b", re.IGNORECASE)),
    ("if-not", re.compile(r"^if\s+not\b", re.IGNORECASE)),
    ("previous-paragraph", re.compile(r"\bprevious\s+paragraph\b", re.IGNORECASE)),
    ("above-fee", re.compile(r"\b(?:the\s+)?above\s+fee\b", re.IGNORECASE)),
    ("above-clauses", re.compile(r"\b(?:the\s+)?above\s+clauses?\b", re.IGNORECASE)),
    ("above-legislation", re.compile(r"\b(?:the\s+)?above\s+legislation\b", re.IGNORECASE)),
    ("above-consideration", re.compile(r"\b(?:the\s+)?consideration\s+above\b", re.IGNORECASE)),
    ("these-amounts", re.compile(r"\bthese\s+amounts?\b", re.IGNORECASE)),
    ("these-thresholds", re.compile(r"\bthese\s+thresholds?\b", re.IGNORECASE)),
    ("these-rights", re.compile(r"\bthese\s+rights?\b", re.IGNORECASE)),
    ("these-banking-activities", re.compile(r"\bthese\s+banking\s+activities\b", re.IGNORECASE)),
    ("such-proceedings", re.compile(r"\bsuch\s+proceedings?\b", re.IGNORECASE)),
    ("pronoun-rights", re.compile(r"^what\s+kind\s+of\s+rights\s+do\s+they\s+have\?", re.IGNORECASE)),
)

FOLLOW_UP_REVIEW_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("what-about", re.compile(r"^what\s+about\b", re.IGNORECASE)),
    ("how-about", re.compile(r"^how\s+about\b", re.IGNORECASE)),
    ("in-view-of-above", re.compile(r"\bin\s+view\s+of\s+the\s+above\b", re.IGNORECASE)),
)

FOLLOW_UP_OVERRIDES: dict[tuple[int, int, int], str] = {}



def classify_follow_up(question: str) -> FollowUpDecision:
    normalized = normalize_extracted_text(question)
    for reason, pattern in FOLLOW_UP_AUTO_PATTERNS:
        if pattern.search(normalized):
            return FollowUpDecision(True, False, reason)
    for reason, pattern in FOLLOW_UP_REVIEW_PATTERNS:
        if pattern.search(normalized):
            return FollowUpDecision(False, True, reason)
    return FollowUpDecision(False, False, None)


def find_profile_start(lines: Sequence[LineRecord], firm: str) -> int | None:
    normalized_firm = normalize_match_text(firm)

    for index, line in enumerate(lines):
        if not normalized_firm:
            break
        normalized = normalize_match_text(line.text)
        if normalized != normalized_firm:
            continue
        lookahead = " ".join(
            normalize_match_text(item.text)
            for item in lines[index + 1:index + 18]
        )
        marker_count = sum(marker.replace("-", " ") in lookahead for marker in PROFILE_MARKERS)
        if marker_count >= 2:
            return index

    for index, line in enumerate(lines):
        text = normalize_extracted_text(line.text)
        if not is_upper_heading(text) or len(text) > 160:
            continue
        lookahead = " ".join(
            normalize_match_text(item.text)
            for item in lines[index + 1:index + 15]
        )
        marker_count = sum(marker.replace("-", " ") in lookahead for marker in PROFILE_MARKERS)
        if marker_count >= 3:
            return index

    return None


def question_groups(lines: Sequence[LineRecord]) -> list[QuestionGroup]:
    groups: list[QuestionGroup] = []

    for index, line in enumerate(lines):
        text = normalize_extracted_text(line.text)
        style = question_style(line)
        if style is None or not text.endswith("?"):
            continue

        start = index
        cursor = index - 1
        while cursor >= 0:
            previous = lines[cursor]
            previous_text = normalize_extracted_text(previous.text)
            if previous.pdf_page != line.pdf_page:
                break
            if previous.block_index != line.block_index:
                break
            if question_style(previous) != style:
                break
            if previous_text.endswith("?"):
                break
            if previous.y1 + 20.0 < lines[cursor + 1].y0:
                break
            start = cursor
            cursor -= 1

        joined = " ".join(
            normalize_extracted_text(lines[item].text)
            for item in range(start, index + 1)
        )
        groups.append(QuestionGroup(start, index, joined))

    deduplicated: list[QuestionGroup] = []
    previous_end = -1
    for group in groups:
        if group.start_index <= previous_end:
            continue
        deduplicated.append(group)
        previous_end = group.end_index
    return deduplicated


def answer_lines(
    lines: Sequence[LineRecord],
    start: int,
    end: int,
    firm: str,
) -> list[LineRecord]:
    values: list[LineRecord] = []
    for line in lines[start:end]:
        text = normalize_extracted_text(line.text)
        if not text:
            continue
        if is_running_matter(line, firm):
            continue
        if is_structural_heading(line):
            continue
        values.append(line)
    return values


def join_answer_text(lines: Sequence[LineRecord]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    current_key: tuple[int, int] | None = None

    for line in lines:
        key = (line.pdf_page, line.block_index)
        text = normalize_extracted_text(line.text)
        if current_key is not None and key != current_key:
            paragraphs.append(" ".join(current))
            current = []
        current.append(text)
        current_key = key

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)


def build_section_samples(
    spec: SectionSpec,
    lines: Sequence[LineRecord],
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    document_id: str = DEFAULT_DOCUMENT_ID,
) -> SectionParseResult:
    filtered = [line for line in lines if not is_running_matter(line, spec.firm)]
    profile_start = find_profile_start(filtered, spec.firm)
    content_end = profile_start if profile_start is not None else len(filtered)
    content = filtered[:content_end]

    warnings: list[str] = []
    blocking: list[str] = []
    review_candidates: list[dict[str, Any]] = []

    embedded_questions = [
        {
            "printed_page": line.printed_page,
            "text": normalize_extracted_text(line.text),
        }
        for line in content
        if is_embedded_question_line(line)
    ]
    if embedded_questions:
        warnings.append(
            f"{len(embedded_questions)} question-mark line(s) were kept inside answer text "
            "because they do not use a top-level GLD question style"
        )

    question_boundaries = question_groups(content)
    if not question_boundaries:
        blocking.append("no styled question boundaries were detected")
        return SectionParseResult([], warnings, blocking, [], review_candidates, embedded_questions)

    pairs: list[QuestionAnswerPair] = []
    pending_questions: list[tuple[int, QuestionGroup]] = []

    for ordinal, group in enumerate(question_boundaries, start=1):
        next_start = (
            question_boundaries[ordinal].start_index
            if ordinal < len(question_boundaries)
            else len(content)
        )
        answer = answer_lines(
            content,
            group.end_index + 1,
            next_start,
            spec.firm,
        )
        pending_questions.append((ordinal, group))

        if not answer:
            continue

        first_ordinal, first_group = pending_questions[0]
        pairs.append(
            QuestionAnswerPair(
                source_ordinals=tuple(
                    source_ordinal
                    for source_ordinal, _ in pending_questions
                ),
                question=" ".join(
                    normalize_extracted_text(question_group.text)
                    for _, question_group in pending_questions
                ),
                answer=join_answer_text(answer),
                page_start=content[first_group.start_index].printed_page,
                page_end=answer[-1].printed_page,
            )
        )
        pending_questions = []

    if pending_questions:
        ordinals = ", ".join(
            str(source_ordinal)
            for source_ordinal, _ in pending_questions
        )
        blocking.append(
            f"{spec.section}: source question(s) {ordinals} have no answer"
        )

    if blocking:
        return SectionParseResult([], warnings, blocking, [], review_candidates, embedded_questions)

    grouped_pairs: list[list[QuestionAnswerPair]] = []
    group_reasons: list[list[str]] = []
    for pair_index, pair in enumerate(pairs):
        first_ordinal = pair.source_ordinals[0]
        override = FOLLOW_UP_OVERRIDES.get(
            (spec.chapter_number, spec.section_number, first_ordinal)
        )
        if override == "merge":
            decision = FollowUpDecision(True, False, "review-override-merge")
        elif override == "standalone":
            decision = FollowUpDecision(False, False, "review-override-standalone")
        elif override is None:
            decision = classify_follow_up(pair.question)
        else:
            raise ValueError(
                f"invalid follow-up override for {spec.section} question {first_ordinal}: "
                f"{override!r}"
            )
        if decision.merge:
            if not grouped_pairs:
                blocking.append(
                    f"{spec.section}: first source question was classified as a follow-up"
                )
                continue
            grouped_pairs[-1].append(pair)
            group_reasons[-1].append(decision.reason or "follow-up")
            continue

        grouped_pairs.append([pair])
        group_reasons.append([])
        if decision.review:
            review_candidates.append(
                {
                    "source_question_ordinals": list(pair.source_ordinals),
                    "question": pair.question,
                    "reason": decision.reason,
                    "previous_source_question_ordinal": (
                        pairs[pair_index - 1].source_ordinals[-1]
                        if pair_index > 0
                        else None
                    ),
                    "previous_question": (
                        pairs[pair_index - 1].question
                        if pair_index > 0
                        else None
                    ),
                }
            )

    if blocking:
        return SectionParseResult([], warnings, blocking, [], review_candidates, embedded_questions)

    samples: list[ReferenceSample] = []
    groups_report: list[dict[str, Any]] = []
    for pair_group, reasons in zip(grouped_pairs, group_reasons):
        first = pair_group[0]
        first_ordinal = first.source_ordinals[0]
        sample_id = (
            f"gld2012-ch{spec.chapter_number:03d}-"
            f"s{spec.section_number:03d}-q{first_ordinal:03d}"
        )
        sample = ReferenceSample(
            schema_version=1,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            sample_id=sample_id,
            chapter=spec.chapter,
            section=spec.section,
            question=" ".join(pair.question for pair in pair_group),
            gold_answer="\n\n".join(pair.answer for pair in pair_group),
            source=ReferenceSource(
                document_id=document_id,
                page_start=first.page_start,
                page_end=pair_group[-1].page_end,
            ),
        )
        samples.append(sample)
        groups_report.append(
            {
                "sample_id": sample_id,
                "source_question_ordinals": [
                    ordinal
                    for pair in pair_group
                    for ordinal in pair.source_ordinals
                ],
                "grouped_follow_up": len(pair_group) > 1,
                "follow_up_reasons": reasons,
                "question": sample.question,
                "page_start": first.page_start,
                "page_end": pair_group[-1].page_end,
            }
        )

    if profile_start is None:
        warnings.append("no firm-profile boundary was detected before the next section")

    return SectionParseResult(
        samples,
        warnings,
        blocking,
        groups_report,
        review_candidates,
        embedded_questions,
    )


def selected_specs(last_included_printed_page: int) -> tuple[list[SectionSpec], SectionSpec]:
    if last_included_printed_page != DEFAULT_LAST_INCLUDED_PRINTED_PAGE:
        raise ValueError(
            "the reviewed GLD 2012 corpus boundary is fixed at printed page 305"
        )

    sentinel = next(
        spec
        for spec in SECTION_SPECS
        if spec.printed_start_page == last_included_printed_page + 1
    )
    selected = [
        spec
        for spec in SECTION_SPECS
        if spec.printed_start_page < sentinel.printed_start_page
    ]
    return selected, sentinel


def verify_pinned_source(pdf_path: Path, page_count: int) -> str:
    source_hash = sha256_file(pdf_path)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "GLD source PDF SHA-256 does not match the pinned thesis copy: "
            f"{source_hash}"
        )
    if page_count != EXPECTED_SOURCE_PAGE_COUNT:
        raise ValueError(
            "GLD source PDF page count does not match the pinned thesis copy: "
            f"{page_count}"
        )
    return source_hash


def _import_pymupdf():
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for GLD import. Install requirements-tools.txt first."
        ) from exc
    return pymupdf


def _page_plain_text(page: Any) -> str:
    return page.get_text("text", sort=True)


def find_anchor_pdf_page(
    document: Any,
    spec: SectionSpec,
    *,
    radius: int = 4,
) -> int:
    predicted_pdf_page = max(1, spec.printed_start_page - 1)
    first = max(1, predicted_pdf_page - radius)
    last = min(document.page_count, predicted_pdf_page + radius)
    needle = normalize_match_text(spec.anchor)

    matches: list[int] = []
    for pdf_page in range(first, last + 1):
        text = normalize_match_text(_page_plain_text(document[pdf_page - 1]))
        if needle in text:
            matches.append(pdf_page)

    if not matches:
        raise ValueError(
            f"could not locate section anchor near printed page {spec.printed_start_page}: "
            f"{spec.anchor!r}"
        )

    return min(matches, key=lambda page: abs(page - predicted_pdf_page))


def extract_pdf_page_lines(page: Any, pdf_page: int, printed_page: int) -> list[LineRecord]:
    payload = page.get_text("dict", sort=True)
    page_height = float(page.rect.height)
    records: list[LineRecord] = []

    for block_index, block in enumerate(payload.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line_index, line in enumerate(block.get("lines", [])):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans)
            text = normalize_extracted_text(text)
            if not text:
                continue

            character_count = 0
            bold_characters = 0
            weighted_size = 0.0
            for span in spans:
                span_text = span.get("text", "")
                count = max(1, len(span_text.strip()))
                character_count += count
                font = str(span.get("font", "")).lower()
                flags = int(span.get("flags", 0))
                if flags & 16 or "bold" in font:
                    bold_characters += count
                weighted_size += float(span.get("size", 0.0)) * count

            bbox = line.get("bbox", (0.0, 0.0, 0.0, 0.0))
            records.append(
                LineRecord(
                    text=text,
                    pdf_page=pdf_page,
                    printed_page=printed_page,
                    block_index=block_index,
                    line_index=line_index,
                    x0=float(bbox[0]),
                    x1=float(bbox[2]),
                    y0=float(bbox[1]),
                    y1=float(bbox[3]),
                    page_height=page_height,
                    bold_ratio=(bold_characters / character_count) if character_count else 0.0,
                    font_size=(weighted_size / character_count) if character_count else 0.0,
                    color=int(max(spans, key=lambda span: len(span.get("text", ""))).get("color", 0)),
                    font_name=str(max(spans, key=lambda span: len(span.get("text", ""))).get("font", "")),
                )
            )
    return records


def extract_section_lines(
    document: Any,
    spec: SectionSpec,
    next_spec: SectionSpec,
) -> tuple[list[LineRecord], dict[str, int]]:
    start_pdf_page = find_anchor_pdf_page(document, spec)
    next_pdf_page = find_anchor_pdf_page(document, next_spec)
    if next_pdf_page <= start_pdf_page:
        raise ValueError(
            f"section boundary is not increasing: {spec.section!r} -> {next_spec.section!r}"
        )

    page_offset = spec.printed_start_page - start_pdf_page
    next_offset = next_spec.printed_start_page - next_pdf_page
    if page_offset != next_offset:
        raise ValueError(
            f"printed/PDF page offset changed between {spec.section!r} and "
            f"{next_spec.section!r}: {page_offset} != {next_offset}"
        )

    records: list[LineRecord] = []
    for pdf_page in range(start_pdf_page, next_pdf_page):
        printed_page = pdf_page + page_offset
        records.extend(
            extract_pdf_page_lines(
                document[pdf_page - 1],
                pdf_page,
                printed_page,
            )
        )

    return records, {
        "start_pdf_page": start_pdf_page,
        "end_pdf_page": next_pdf_page - 1,
        "start_printed_page": spec.printed_start_page,
        "end_printed_page": next_spec.printed_start_page - 1,
        "page_offset": page_offset,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_candidate(output_dir: Path, samples: Iterable[ReferenceSample]) -> None:
    values = list(samples)
    if not values:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = output_dir / "candidate_all.jsonl"
    pretty = output_dir / "candidate_all.pretty.json"
    write_reference_jsonl(jsonl, values)
    write_pretty_reference_json(jsonl, pretty)


def import_gld(
    pdf_path: Path,
    output_dir: Path,
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    document_id: str = DEFAULT_DOCUMENT_ID,
    last_included_printed_page: int = DEFAULT_LAST_INCLUDED_PRINTED_PAGE,
) -> dict[str, Any]:
    pymupdf = _import_pymupdf()
    selected, sentinel = selected_specs(last_included_printed_page)
    ordered = selected + [sentinel]

    all_samples: list[ReferenceSample] = []
    review_sections: list[dict[str, Any]] = []
    blocking_issues: list[str] = []
    warnings: list[str] = []

    with pymupdf.open(pdf_path) as document:
        source_hash = verify_pinned_source(pdf_path, document.page_count)
        for index, spec in enumerate(selected):
            next_spec = ordered[index + 1]
            if not spec.qa:
                review_sections.append(
                    {
                        "chapter": spec.chapter,
                        "section": spec.section,
                        "printed_start_page": spec.printed_start_page,
                        "sample_count": 0,
                        "status": "excluded_non_qa",
                        "boundary": {
                            "start_printed_page": spec.printed_start_page,
                            "end_printed_page": next_spec.printed_start_page - 1,
                        },
                        "warnings": [
                            "section is prose-structured in the pinned GLD edition and is excluded "
                            "because LegalFedLLM does not synthesize questions"
                        ],
                        "blocking_issues": [],
                        "groups": [],
                        "follow_up_review_candidates": [],
                        "embedded_question_lines": [],
                    }
                )
                continue
            try:
                lines, boundary = extract_section_lines(document, spec, next_spec)
                parsed = build_section_samples(
                    spec,
                    lines,
                    dataset_id=dataset_id,
                    dataset_version=dataset_version,
                    document_id=document_id,
                )
            except Exception as exc:
                message = f"{spec.section}: {exc}"
                blocking_issues.append(message)
                review_sections.append(
                    {
                        "chapter": spec.chapter,
                        "section": spec.section,
                        "printed_start_page": spec.printed_start_page,
                        "sample_count": 0,
                        "warnings": [],
                        "blocking_issues": [str(exc)],
                    }
                )
                continue

            all_samples.extend(parsed.samples)
            section_blocking = [
                f"{spec.section}: {message}"
                for message in parsed.blocking_issues
            ]
            section_warnings = [
                f"{spec.section}: {message}"
                for message in parsed.warnings
            ]
            blocking_issues.extend(section_blocking)
            warnings.extend(section_warnings)
            review_sections.append(
                {
                    "chapter": spec.chapter,
                    "section": spec.section,
                    "printed_start_page": spec.printed_start_page,
                    "sample_count": len(parsed.samples),
                    "boundary": boundary,
                    "warnings": parsed.warnings,
                    "blocking_issues": parsed.blocking_issues,
                    "groups": parsed.groups,
                    "follow_up_review_candidates": parsed.review_candidates,
                    "embedded_question_lines": parsed.embedded_question_lines,
                }
            )

        source_page_count = document.page_count

    output_dir.mkdir(parents=True, exist_ok=True)
    follow_up_review_candidate_count = sum(
        len(section.get("follow_up_review_candidates", []))
        for section in review_sections
    )
    review = {
        "status": (
            "clean"
            if not blocking_issues and follow_up_review_candidate_count == 0
            else "needs_review"
        ),
        "source": {
            "filename": pdf_path.name,
            "sha256": source_hash,
            "page_count": source_page_count,
        },
        "extractor": {
            "name": "PyMuPDF",
            "version": getattr(pymupdf, "VersionBind", "unknown"),
            "text_sort": True,
        },
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "scope": {
            "first_section_start_printed_page": selected[0].printed_start_page,
            "last_section_start_printed_page": selected[-1].printed_start_page,
            "last_included_printed_page": sentinel.printed_start_page - 1,
            "next_section_start_printed_page": sentinel.printed_start_page,
            "section_count": len(selected),
            "qa_section_count": sum(1 for spec in selected if spec.qa),
            "excluded_non_qa_section_count": sum(1 for spec in selected if not spec.qa),
        },
        "sample_count": len(all_samples),
        "follow_up_review_candidate_count": follow_up_review_candidate_count,
        "warnings": warnings,
        "blocking_issues": blocking_issues,
        "sections": review_sections,
    }
    write_json(output_dir / "review.json", review)

    if blocking_issues or follow_up_review_candidate_count:
        write_candidate(output_dir, all_samples)
        raise RuntimeError(
            "GLD import needs review: "
            f"{len(blocking_issues)} blocking issue(s), "
            f"{follow_up_review_candidate_count} follow-up candidate(s). "
            f"Inspect {output_dir / 'review.json'} and candidate_all.pretty.json."
        )

    reference, validation = split_reference_samples(all_samples)

    paths = {
        "all": output_dir / "all.jsonl",
        "reference": output_dir / "reference.jsonl",
        "validation": output_dir / "validation.jsonl",
    }
    for key, path in paths.items():
        values = {
            "all": all_samples,
            "reference": reference,
            "validation": validation,
        }[key]
        write_reference_jsonl(path, values)
        write_pretty_reference_json(
            path,
            output_dir / f"{key}.pretty.json",
        )

    identity = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "source": {
            "document_id": document_id,
            "filename": pdf_path.name,
            "sha256": source_hash,
            "page_count": source_page_count,
        },
        "extractor": review["extractor"],
        "scope": review["scope"],
        "all": reference_dataset_identity(all_samples).model_dump(mode="json"),
        "reference": reference_dataset_identity(reference).model_dump(mode="json"),
        "validation": reference_dataset_identity(validation).model_dump(mode="json"),
    }
    write_json(output_dir / "identity.json", identity)
    return identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import the initial Greek Law Digest 2012 Q&A corpus into LegalFedLLM JSONL."
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--document-id", default=DEFAULT_DOCUMENT_ID)
    parser.add_argument(
        "--last-included-printed-page",
        type=int,
        default=DEFAULT_LAST_INCLUDED_PRINTED_PAGE,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.pdf.is_file():
        print(f"GLD PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    try:
        identity = import_gld(
            args.pdf,
            args.output_dir,
            dataset_id=args.dataset_id,
            dataset_version=args.dataset_version,
            document_id=args.document_id,
            last_included_printed_page=args.last_included_printed_page,
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(identity, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
