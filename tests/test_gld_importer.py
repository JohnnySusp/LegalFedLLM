from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.datasets.gld_pdf_to_jsonl import (
    BLUE_TEXT_COLOR,
    DEFAULT_DATASET_ID,
    DEFAULT_LAST_INCLUDED_PRINTED_PAGE,
    EXPECTED_SOURCE_PAGE_COUNT,
    EXPECTED_SOURCE_SHA256,
    FOLLOW_UP_OVERRIDES,
    LineRecord,
    SectionSpec,
    audit_corpus_samples,
    build_section_samples,
    classify_follow_up,
    is_embedded_question_line,
    is_primary_question_style,
    is_running_matter,
    is_secondary_question_style,
    normalize_extracted_text,
    selected_specs,
    verify_pinned_source,
)


def line(
    text: str,
    *,
    printed_page: int = 34,
    pdf_page: int = 33,
    block: int = 1,
    index: int = 0,
    style: str = "regular",
    y0: float = 100.0,
    y1: float = 110.0,
) -> LineRecord:
    if style == "primary":
        x0 = 42.5
        bold = 1.0
        size = 10.0
        color = BLUE_TEXT_COLOR
        font = "MyriadPro-Semibold"
    elif style == "secondary":
        x0 = 65.2
        bold = 1.0
        size = 9.8
        color = 0
        font = "MyriadPro-Bold"
    else:
        x0 = 70.9
        bold = 0.0
        size = 10.0
        color = 0
        font = "MyriadPro-Regular"

    return LineRecord(
        text=text,
        pdf_page=pdf_page,
        printed_page=printed_page,
        block_index=block,
        line_index=index,
        x0=x0,
        x1=400.0,
        y0=y0,
        y1=y1,
        page_height=842.0,
        bold_ratio=bold,
        font_size=size,
        color=color,
        font_name=font,
    )


SPEC = SectionSpec(
    chapter_number=1,
    section_number=1,
    chapter="JUDICIAL SYSTEM",
    section="PROCEDURE BEFORE CIVIL COURTS",
    printed_start_page=34,
    firm="Kelemenis & Co.",
    anchor="PROCEDURE BEFORE CIVIL COURTS",
)

CONTRACT_SPEC = SectionSpec(
    chapter_number=2,
    section_number=1,
    chapter="BASIC ASPECTS OF CIVIL LAW",
    section="GENERAL PRINCIPLES OF CONTRACT LAW",
    printed_start_page=76,
    firm="A.S. Papadimitriou & Partners Law Firm",
    anchor="GENERAL PRINCIPLES OF CONTRACT LAW",
)

MINORITY_SPEC = SectionSpec(
    chapter_number=3,
    section_number=3,
    chapter="BUSINESS ENTITIES",
    section=(
        "SOCIETE ANONYME - COMPANY LIMITED BY SHARES / "
        "Minority Shareholders Rights - Shareholders Agreements"
    ),
    printed_start_page=124,
    firm="Dryllerakis & Associates",
    anchor="Minority Shareholders Rights - Shareholders Agreements",
)

MUTUAL_SPEC = SectionSpec(
    chapter_number=3,
    section_number=11,
    chapter="BUSINESS ENTITIES",
    section="MUTUAL FUNDS - PORTFOLIO INVESTMENT COMPANIES - VENTURE CAPITAL",
    printed_start_page=167,
    firm="A.S. Papadimitriou & Partners Law Firm",
    anchor="MUTUAL FUNDS - PORTFOLIO INVESTMENT COMPANIES - VENTURE CAPITAL",
)


class GldImporterTests(unittest.TestCase):
    def test_primary_and_secondary_question_styles_are_recognized(self) -> None:
        self.assertTrue(
            is_primary_question_style(
                line("What rules apply?", style="primary")
            )
        )
        self.assertTrue(
            is_secondary_question_style(
                line("What remedies exist?", style="secondary")
            )
        )

    def test_multiline_primary_question_is_joined(self) -> None:
        lines = [
            line(
                "How is the Greek civil court",
                style="primary",
                index=0,
                y0=120,
                y1=130,
            ),
            line(
                "system structured?",
                style="primary",
                index=1,
                y0=129.8,
                y1=140,
            ),
            line("It has several levels.", index=2, y0=145, y1=155),
        ]

        result = build_section_samples(SPEC, lines)

        self.assertEqual(result.blocking_issues, [])
        self.assertEqual(len(result.samples), 1)
        self.assertEqual(
            result.samples[0].question,
            "How is the Greek civil court system structured?",
        )

    def test_blue_caption_without_question_mark_is_not_a_question(self) -> None:
        lines = [
            line("Introduction", style="primary", block=1, index=0),
            line("What rules apply?", style="primary", block=2, index=0),
            line("ADVERTISEMENT CONTENT", style="primary", block=3, index=0),
            line("The substantive answer.", block=4, index=0),
        ]

        result = build_section_samples(SPEC, lines)

        self.assertEqual(len(result.samples), 1)
        self.assertEqual(result.samples[0].question, "What rules apply?")
        self.assertEqual(result.samples[0].gold_answer, "The substantive answer.")

    def test_black_bold_question_is_a_top_level_question(self) -> None:
        lines = [
            line("12. Legal Remedies", style="primary", index=0),
            line(
                "What are the ordinary legal remedies?",
                style="secondary",
                block=2,
                index=0,
            ),
            line("The defendant may appeal.", block=3, index=0),
        ]

        result = build_section_samples(SPEC, lines)

        self.assertEqual(len(result.samples), 1)
        self.assertEqual(
            result.samples[0].question,
            "What are the ordinary legal remedies?",
        )

    def test_black_bold_answer_caption_is_preserved(self) -> None:
        lines = [
            line("What tax rules apply?", style="primary", block=1),
            line("Income tax (Capital gains tax)", style="secondary", block=2),
            line("A capital gains rule applies.", block=3),
            line("VAT and Stamp duty", style="secondary", block=4),
            line("A separate VAT rule applies.", block=5),
        ]

        result = build_section_samples(SPEC, lines)

        self.assertEqual(len(result.samples), 1)
        self.assertIn(
            "Income tax (Capital gains tax)",
            result.samples[0].gold_answer,
        )
        self.assertIn(
            "VAT and Stamp duty",
            result.samples[0].gold_answer,
        )

    def test_regular_question_inside_answer_is_preserved_as_answer_text(self) -> None:
        embedded = line(
            "What follows if negotiation efforts succeed and result to an agreement?",
            style="regular",
            index=2,
        )
        self.assertTrue(is_embedded_question_line(embedded))

        lines = [
            line("When does conciliation take place?", style="primary", index=0),
            line("The parties first negotiate.", index=1),
            embedded,
            line("The agreement is recorded in writing.", index=3),
        ]

        result = build_section_samples(SPEC, lines)

        self.assertEqual(len(result.samples), 1)
        self.assertIn(
            "What follows if negotiation efforts succeed and result to an agreement?",
            result.samples[0].gold_answer,
        )
        self.assertTrue(result.warnings)

    def test_consecutive_questions_with_one_answer_are_one_source_pair(self) -> None:
        lines = [
            line(
                "What if the other party fails to abide with contractual obligations other than performance?",
                style="primary",
                block=1,
                index=0,
            ),
            line(
                "What remedies are there?",
                style="primary",
                block=2,
                index=0,
            ),
            line(
                "The injured party may seek compensation.",
                block=3,
                index=0,
            ),
        ]

        result = build_section_samples(SPEC, lines)

        self.assertEqual(result.blocking_issues, [])
        self.assertEqual(len(result.samples), 1)
        self.assertEqual(
            result.samples[0].question,
            "What if the other party fails to abide with contractual obligations other than performance? What remedies are there?",
        )
        self.assertEqual(
            result.groups[0]["source_question_ordinals"],
            [1, 2],
        )

    def test_confirmed_follow_up_is_appended_to_previous_sample(self) -> None:
        lines = [
            line(
                "Can the seller secure payment by a reservation-of-title clause?",
                style="primary",
                block=1,
            ),
            line("Yes, subject to the applicable rules.", block=2),
            line(
                "But what, if the buyer transfers the goods to third parties?",
                style="primary",
                block=3,
            ),
            line("The third-party transfer has separate effects.", block=4),
            line(
                "What applies in case of insolvency?",
                style="primary",
                block=5,
            ),
            line("Insolvency rules then apply.", block=6),
        ]

        result = build_section_samples(SPEC, lines)

        self.assertEqual(len(result.samples), 2)
        self.assertEqual(
            result.samples[0].question,
            "Can the seller secure payment by a reservation-of-title clause? But what, if the buyer transfers the goods to third parties?",
        )
        self.assertEqual(
            result.samples[0].gold_answer,
            "Yes, subject to the applicable rules.\n\nThe third-party transfer has separate effects.",
        )
        self.assertEqual(
            [sample.sample_id for sample in result.samples],
            [
                "gld2012-ch001-s001-q001",
                "gld2012-ch001-s001-q003",
            ],
        )
        self.assertEqual(
            result.groups[0]["source_question_ordinals"],
            [1, 2],
        )
        self.assertTrue(result.groups[0]["grouped_follow_up"])

    def test_reviewed_implied_terms_question_is_standalone(self) -> None:
        self.assertEqual(FOLLOW_UP_OVERRIDES[(2, 1, 11)], "standalone")

        lines = []
        for ordinal in range(1, 11):
            lines.extend(
                [
                    line(
                        f"What is placeholder contract question {ordinal}?",
                        style="primary",
                        block=ordinal * 2,
                    ),
                    line(
                        f"Placeholder contract answer {ordinal}.",
                        block=ordinal * 2 + 1,
                    ),
                ]
            )
        lines.extend(
            [
                line(
                    "What about implied terms? Can they be accepted as part of a contract?",
                    style="primary",
                    block=22,
                ),
                line("Implied terms may be accepted.", block=23),
            ]
        )

        result = build_section_samples(CONTRACT_SPEC, lines)

        self.assertEqual(len(result.samples), 11)
        self.assertEqual(result.review_candidates, [])

    def test_reviewed_shareholders_question_is_merged(self) -> None:
        self.assertEqual(FOLLOW_UP_OVERRIDES[(3, 3, 21)], "merge")

        lines = []
        for ordinal in range(1, 20):
            lines.extend(
                [
                    line(
                        f"What is placeholder question {ordinal}?",
                        style="primary",
                        block=ordinal * 2,
                    ),
                    line(
                        f"Placeholder answer {ordinal}.",
                        block=ordinal * 2 + 1,
                    ),
                ]
            )
        lines.extend(
            [
                line(
                    "What happens if there is a conflict between the Articles of Association and a Shareholders Agreement?",
                    style="primary",
                    block=40,
                ),
                line("The documents have different legal effects.", block=41),
                line(
                    "In view of the above how can a Shareholders Agreement be enforced?",
                    style="primary",
                    block=42,
                ),
                line("It may be enforced through courts or arbitration.", block=43),
            ]
        )

        result = build_section_samples(MINORITY_SPEC, lines)

        self.assertEqual(result.review_candidates, [])
        self.assertEqual(len(result.samples), 20)
        self.assertEqual(
            result.samples[-1].question,
            "What happens if there is a conflict between the Articles of Association and a Shareholders Agreement? In view of the above how can a Shareholders Agreement be enforced?",
        )
        self.assertEqual(
            result.groups[-1]["source_question_ordinals"],
            [20, 21],
        )

    def test_approved_subsection_caption_disambiguates_repeated_questions(self) -> None:
        lines = [
            line("MUTUAL FUNDS", style="primary", block=1),
            line("How are units acquired by the unit holders?", style="primary", block=2),
            line("The MF has an acquisition procedure.", block=3),
            line("What about reporting requirements?", style="primary", block=4),
            line("The MF has reporting requirements.", block=5),
            line("PORTFOLIO INVESTMENT COMPANIES", style="primary", block=6),
            line("How are PIC’s shares acquired?", style="primary", block=7),
            line("The PIC has a listing procedure.", block=8),
            line("What about reporting requirements?", style="primary", block=9),
            line("The PIC has reporting requirements.", block=10),
        ]

        result = build_section_samples(MUTUAL_SPEC, lines)

        self.assertEqual(len(result.samples), 4)
        self.assertEqual(
            result.samples[1].section,
            f"{MUTUAL_SPEC.section} / MUTUAL FUNDS",
        )
        self.assertEqual(
            result.samples[3].section,
            f"{MUTUAL_SPEC.section} / PORTFOLIO INVESTMENT COMPANIES",
        )
        audit = audit_corpus_samples(result.samples)
        self.assertEqual(audit["duplicate_prompts"], [])

    def test_contributing_firm_running_footer_is_removed(self) -> None:
        footer = line(
            "Kelemenis & Co.",
            style="regular",
            y0=790.0,
            y1=810.0,
        )
        self.assertTrue(is_running_matter(footer, SPEC.firm))

        lines = [
            line("What rules apply?", style="primary", block=1),
            line("The substantive answer.", block=2),
            footer,
        ]

        result = build_section_samples(SPEC, lines)

        self.assertEqual(len(result.samples), 1)
        self.assertNotIn("Kelemenis", result.samples[0].gold_answer)

    def test_corpus_audit_detects_duplicate_canonical_prompts(self) -> None:
        lines = [
            line("What rules apply?", style="primary", block=1),
            line("The answer.", block=2),
        ]
        sample = build_section_samples(SPEC, lines).samples[0]

        audit = audit_corpus_samples([sample, sample])

        self.assertEqual(audit["status"], "needs_review")
        self.assertEqual(len(audit["duplicate_prompts"]), 1)

    def test_ambiguous_follow_up_without_reviewed_override_is_reported(self) -> None:
        decision = classify_follow_up("What about reporting requirements?")
        self.assertFalse(decision.merge)
        self.assertTrue(decision.review)

        lines = [
            line("How are units acquired?", style="primary", block=1),
            line("Units are acquired under the applicable procedure.", block=2),
            line("What about reporting requirements?", style="primary", block=3),
            line("Reporting requirements also apply.", block=4),
        ]

        result = build_section_samples(SPEC, lines)

        self.assertEqual(len(result.samples), 2)
        self.assertEqual(len(result.review_candidates), 1)
        self.assertEqual(
            result.review_candidates[0]["source_question_ordinals"],
            [2],
        )

    def test_extraction_hyphen_marker_is_normalized(self) -> None:
        self.assertEqual(
            normalize_extracted_text("cross\ufffeborder rules"),
            "cross-border rules",
        )

    def test_scope_is_fixed_to_printed_pages_34_through_305(self) -> None:
        selected, sentinel = selected_specs(DEFAULT_LAST_INCLUDED_PRINTED_PAGE)

        self.assertEqual(len(selected), 46)
        self.assertEqual(sum(1 for spec in selected if spec.qa), 44)
        self.assertEqual(selected[0].printed_start_page, 34)
        self.assertEqual(selected[-1].printed_start_page, 300)
        self.assertEqual(sentinel.printed_start_page, 306)
        self.assertEqual(
            [spec.printed_start_page for spec in selected if not spec.qa],
            [148, 163],
        )

        with self.assertRaises(ValueError):
            selected_specs(306)

    def test_pinned_thesis_source_identity_is_fixed(self) -> None:
        self.assertEqual(EXPECTED_SOURCE_PAGE_COUNT, 713)
        self.assertEqual(
            EXPECTED_SOURCE_SHA256,
            "9673ee7c86b3d582e2c08e1cdd2b84f144981f31a1fe50d4216e82c5b350b77d",
        )

        with tempfile.TemporaryDirectory() as directory:
            fake_pdf = Path(directory) / "different.pdf"
            fake_pdf.write_bytes(b"not the pinned GLD")
            with self.assertRaises(ValueError):
                verify_pinned_source(fake_pdf, EXPECTED_SOURCE_PAGE_COUNT)

    def test_dataset_id_remains_generic_runtime_input(self) -> None:
        lines = [
            line("What rules apply?", style="primary", block=1),
            line("The answer.", block=2),
        ]

        result = build_section_samples(SPEC, lines)

        self.assertEqual(result.samples[0].dataset_id, DEFAULT_DATASET_ID)


if __name__ == "__main__":
    unittest.main(verbosity=2)
