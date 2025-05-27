from ..clause_configs.clause_config_ordinary_course import ORDINARY_COURSE_CLAUSE_CONFIG
from ..clause_configs.clause_config_non_solicitation import (
    NON_SOLICITATION_CLAUSE_CONFIG,
)
from .summarize_util import SummaryEngineUtil


class ClauseConfigUtil:
    def __init__(self):
        clause_config = {
            **ORDINARY_COURSE_CLAUSE_CONFIG,
            **NON_SOLICITATION_CLAUSE_CONFIG,
            # Add more configs here
        }

        self.clause_config = clause_config
        self.active_clauses = {
            "non_solicitation_clauses": [
                "Match Right",
                "Go Shop",
                "No Shop Clause",
                "Notice of Competing Offer",
                "Ongoing Update",
                "Superior Proposal Engagement",
                "Adverse Recommendation (Superior Proposal)",
                "Adverse Recommendation (Intervening Event)",
            ],
            "convenant_clauses": [
                "Ordinary Course",
            ],
        }

    def get_organized_sections_for_summary(self, schema):

        concise = []
        fulsome = []

        # =========================
        # ORDER OVERRIDES (Optional)
        # =========================
        CONCISE_ORDER = []
        FULSOME_ORDER = []

        all_labels = self.clause_config.keys()

        sumamry_util = SummaryEngineUtil()
        sections = []

        for category in self.active_clauses:
            section_obj = {category: {"concise": {}, "fullsome": {}}}
            sections.append(section_obj)

            for base in self.active_clauses[category]:
                concise_label = f"{base} - Concise"
                fulsome_label = f"{base} - Fulsome"

                if concise_label in all_labels:
                    summary = sumamry_util.summarize_from_config(
                        schema, self.clause_config[concise_label]
                    )
                    section_obj[category]["concise"][concise_label] = summary

                if fulsome_label in all_labels:
                    summary = sumamry_util.summarize_from_config(
                        schema, self.clause_config[fulsome_label]
                    )
                    section_obj[category]["fullsome"][fulsome_label] = summary

        return sections

        # def get_organized_sections_for_summary(self, schema):

        concise = []
        fulsome = []

        # =========================
        # ORDER OVERRIDES (Optional)
        # =========================
        CONCISE_ORDER = []
        FULSOME_ORDER = []

        all_labels = self.clause_config.keys()

        for base in self.active_clauses:
            concise_label = f"{base} - Concise"
            fulsome_label = f"{base} - Fulsome"

            if concise_label in all_labels:
                concise.append(concise_label)
            if fulsome_label in all_labels:
                fulsome.append(fulsome_label)

        # Sort logic: apply manual order if provided, else alphabetical
        concise = (
            sorted(concise)
            if not CONCISE_ORDER
            else [c for c in CONCISE_ORDER if c in concise]
        )
        fulsome = (
            sorted(fulsome)
            if not FULSOME_ORDER
            else [f for f in FULSOME_ORDER if f in fulsome]
        )

        sections = []

        sumamry_util = SummaryEngineUtil()

        if concise:
            for label in concise:
                summary = sumamry_util.summarize_from_config(
                    schema, self.clause_config[label]
                )
                sections.append((label, summary))
        else:
            print(f"No Concise available for sections")

        if fulsome:
            if concise:
                sections.append(("\u00a0", None))  # blank line between
            sections.append(("=== Fulsome Summary ===", None))
            for label in fulsome:
                summary = sumamry_util.summarize_from_config(
                    schema, self.clause_config[label]
                )
                sections.append((label, summary))

        else:
            print(f"No fulsome available for sections")

        return sections
