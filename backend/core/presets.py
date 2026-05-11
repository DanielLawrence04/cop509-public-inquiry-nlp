"""Preset document pairs — maps UI preset IDs to PDF paths in data/raw/.

Each preset belongs to a ``dataset_group``:
  * ``coursework_given`` — the official document pairs supplied with the
    COP509 coursework dataset.
  * ``extra_found`` — additional pairs sourced beyond the coursework set,
    used to test pipeline generalisation. They are clearly marked in the UI.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


DATASET_GROUPS: dict[str, dict[str, str]] = {
    "coursework_given": {
        "label": "Coursework given documents",
        "description": "Official document pairs supplied with the COP509 coursework dataset.",
    },
    "extra_found": {
        "label": "Extra documents found for extension/testing",
        "description": "Extra document pair — added to test generalisation beyond the coursework dataset.",
    },
}


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    policy_pdf: Path
    response_pdf: Path
    dataset_group: str = "coursework_given"
    # When true, if recommendation extraction from ``policy_pdf`` returns zero
    # rows the pipeline falls back to extracting "Recommendation N:" headings
    # from ``response_pdf``. Used for documents (e.g. Grenfell Phase 2 Vol 7)
    # whose recommendation PDF does not expose a clean numbered list, but
    # whose government response quotes each recommendation verbatim before
    # responding.
    allow_response_heading_recommendation_fallback: bool = False
    # When set (e.g. "113" for Grenfell Phase 2), triggers an inline-paragraph
    # recommendation extractor that walks paragraphs numbered "<prefix>.N" and
    # captures the recommendation clause embedded in prose ("We therefore
    # recommend that …", "We recommend that …", "We also recommend …").
    # Used when the recommendation PDF doesn't expose a clean numbered list.
    inline_recommendation_chapter_prefix: str | None = None
    # When true, treat the policy PDF as a select-committee report and extract
    # recommendations only from numbered items in the final
    # "Conclusions and recommendations" section that contain a directive
    # recommendation phrase (e.g. "we recommend that …",
    # "the Government should …"). Conclusion items in the same numbered list
    # are excluded. Used for House of Commons Home Affairs Committee style
    # reports such as "Police response to the 2024 summer disorder".
    select_committee_conclusions_section: bool = False

    @property
    def group_label(self) -> str:
        return DATASET_GROUPS[self.dataset_group]["label"]

    @property
    def group_description(self) -> str:
        return DATASET_GROUPS[self.dataset_group]["description"]

    @property
    def is_extra(self) -> bool:
        return self.dataset_group == "extra_found"


PRESETS: dict[str, Preset] = {
    p.id: p for p in [
        # ── Coursework-given pairs ────────────────────────────────────────
        Preset(
            id="behaviour_change",
            label="Behaviour Change",
            policy_pdf=_RAW / "Behaviour-Change-Report-Recomm.pdf",
            response_pdf=_RAW / "Behaviour-Change-Response.pdf",
            dataset_group="coursework_given",
        ),
        Preset(
            id="post_office",
            label="Post Office Horizon Inquiry",
            policy_pdf=_RAW / "PostOfficeHorizon-I- Inquiry-Recomm.pdf",
            response_pdf=_RAW / "PostOfficeHorizon-IT-Inquiry-Response.pdf",
            dataset_group="coursework_given",
        ),
        Preset(
            id="space_economy",
            label="The Space Economy",
            policy_pdf=_RAW / "TheSpaceEconomyReport.pdf",
            response_pdf=_RAW / "TheSpaceEconomyResponse.pdf",
            dataset_group="coursework_given",
        ),
        Preset(
            id="covid_inquiry",
            label="UK Covid-19 Inquiry Module 1",
            policy_pdf=_RAW / "UK-Covid-19-Inquiry-Module-1-Recomm.pdf",
            response_pdf=_RAW / "UK-Covid-19_Inquiry_Module_1_Response.pdf",
            dataset_group="coursework_given",
        ),
        Preset(
            id="blood_inquiry",
            label="Infected Blood Inquiry",
            policy_pdf=_RAW / "Volume_1-Blood-Inquiry-Recomm.pdf",
            response_pdf=_RAW / "Volume_1-Blood-Inquiry-Response.pdf",
            dataset_group="coursework_given",
        ),
        # ── Extra pairs sourced beyond the coursework dataset ─────────────
        Preset(
            id="grenfell_phase2",
            label="Grenfell Tower Inquiry — Phase 2",
            policy_pdf=_RAW / "Grenfell-Phase2-Volume7-Recomm.pdf",
            response_pdf=_RAW / "Grenfell-Phase2-Response.pdf",
            dataset_group="extra_found",
            allow_response_heading_recommendation_fallback=True,
            inline_recommendation_chapter_prefix="113",
        ),
        Preset(
            id="covid_inquiry_module2",
            label="UK Covid-19 Inquiry Module 2",
            policy_pdf=_RAW / "UK-Covid-19-Inquiry-Module-2-Recomm.pdf",
            response_pdf=_RAW / "UK-Covid-19_Inquiry_Module_2_Response.pdf",
            dataset_group="extra_found",
        ),
        Preset(
            id="summer_2024_disorder",
            label="Police response to the 2024 summer disorder",
            policy_pdf=_RAW / "Summer2024-Disorder-Recomm.pdf",
            response_pdf=_RAW / "Summer2024-Disorder-Response.pdf",
            dataset_group="extra_found",
            select_committee_conclusions_section=True,
        ),
    ]
}


def validate_preset_files(preset: Preset) -> None:
    """Raise FileNotFoundError naming the missing PDF(s) for clarity."""
    missing: list[str] = []
    if not preset.policy_pdf.exists():
        missing.append(f"recommendation PDF: {preset.policy_pdf.name}")
    if not preset.response_pdf.exists():
        missing.append(f"response PDF: {preset.response_pdf.name}")
    if missing:
        raise FileNotFoundError(
            f"Preset '{preset.id}' is missing expected file(s): " + "; ".join(missing)
        )
