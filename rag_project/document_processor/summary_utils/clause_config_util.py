from ..clause_configs.clause_config_ordinary_course import ORDINARY_COURSE_CLAUSE_CONFIG
from ..clause_configs.clause_config_non_solicitation import (
    NON_SOLICITATION_CLAUSE_CONFIG,
)
from ..clause_configs.clause_config_conditions_to_closing import (
    CLOSING_CONDITIONS_CLAUSE_CONFIG,
)
from ..clause_configs.financing_summary import (
    FINANCING_SUMMARY_CLAUSE_CONFIG)
from ..clause_configs.closing_mechanics import (
    CLOSING_MECHANICS_CLAUSE_CONFIG,
)
from .summarize_util import SummaryEngineUtil
import logging
from .conditions_summary import ConditionSummarizer
from .regulatory_best_efforts_summary import RegulatorySummarizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from .board_approvals_summary import BoardApprovalsSummarizer
logger = logging.getLogger(__name__)


class ClauseConfigUtil:
    def __init__(self):
        clause_config = {
            **ORDINARY_COURSE_CLAUSE_CONFIG,
            **NON_SOLICITATION_CLAUSE_CONFIG,
            **CLOSING_CONDITIONS_CLAUSE_CONFIG,
            **FINANCING_SUMMARY_CLAUSE_CONFIG,
            **CLOSING_MECHANICS_CLAUSE_CONFIG,
            # Add more configs here
        }

        self.clause_config = clause_config
        self.active_clauses = {

            "best_efforts": [],
            "board_approvals": [],
            "conditions_to_closing": [],
            "convenant_clauses": [
                "Ordinary Course",
            ],
            "closing_mechanics": [
                "Target Date",
                "Marketing Period",
                "Inside Date"
            ],
            "financial_clauses": [
                "Committed Financing",
                "Cash On Hand Financing",
                "Financing Efforts Summary",
                "Substitute Financing Notice And Efforts",


            ],
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




        }

    def get_organized_sections_for_summary(self, schema):
        sections = []
        all_labels = self.clause_config.keys()
        sumamry_util = SummaryEngineUtil()
        print(f"DEBUG: all_labels → {all_labels}")

        def summarize_clause(category, base):
            concise_label = f"{base} - Concise"
            fulsome_label = f"{base} - Fulsome"
            result = {"category": category, "concise": {}, "fullsome": {}}

            if concise_label in all_labels:
                summary = sumamry_util.summarize_from_config(
                    schema, self.clause_config[concise_label])
                result["concise"][concise_label] = summary

            if fulsome_label in all_labels:
                summary = sumamry_util.summarize_from_config(
                    schema, self.clause_config[fulsome_label])
                result["fullsome"][fulsome_label] = summary

            return result

        with ThreadPoolExecutor(max_workers=25) as executor:
            # Store (future, section_obj, clause_type, label)
            futures_meta = []

            for category in self.active_clauses:
                section_obj = {category: {"concise": {}, "fullsome": {}}}
                sections.append(section_obj)

                if category == "best_efforts":
                    summarizer = RegulatorySummarizer(schema.get(
                        "best_efforts"), parent_name="James Hardie")

                    # Submit tasks
                    task_map = {
                        "Divestiture – Concise": summarizer.summarize_divestiture_section,
                        "Litigation – Concise": summarizer.summarize_litigation_commitments,
                        "HSR Filing – fullsome": summarizer.summarize_hsr_filing,
                        "CFIUS Filing – fullsome": summarizer.summarize_cfiius_filing,
                        "Foreign Filing – fullsome": summarizer.summarize_foreign_filing,
                        "Effort Standard – fullsome": summarizer.summarize_effort_standard,
                        "Withdrawal Control – fullsome": summarizer.summarize_withdrawal_controls,
                        "Timing Agreement Restriction – fullsome": summarizer.summarize_timing_agreement,
                        "Divestiture Cap Fulfsome": summarizer.summarize_divestiture_cap_fulsome,
                        "Prior Approval Commitment – fullsome": summarizer.summarize_prior_approval_commitment,
                        "Transaction Interference – fullsome": summarizer.summarize_transaction_interference,
                        "Second Request Certification – fullsome": summarizer.summarize_second_request_certification,
                        "FTC Warning Letter Handling – fullsome": summarizer.summarize_ftc_warning_letter_handling,
                    }

                    for label, fn in task_map.items():
                        clause_type = "concise" if "Concise" in label else "fullsome"
                        future = executor.submit(fn)
                        futures_meta.append(
                            (future, section_obj, category, clause_type, label))

                elif category == "board_approvals":
                    summarizer = BoardApprovalsSummarizer(
                        schema.get("board_approval"))
                    concise_future = executor.submit(
                        summarizer.get_consice_summary)
                    fullsome_future = executor.submit(
                        summarizer.get_fulsome_summary)

                    futures_meta.append(
                        (concise_future, section_obj, category, "concise", "Board Approvals – Concise"))
                    futures_meta.append(
                        (fullsome_future, section_obj, category, "fullsome", "Board Approvals – fullsome"))

                elif category == "conditions_to_closing":
                    summarizer = ConditionSummarizer(
                        schema.get("conditions_to_closing"))

                    task_map = {
                        "Mutual Conditions – Concise": summarizer.summarize_mutual_conditions,
                        "Parent Closing Conditions – Concise": summarizer.summarize_parent_conditions,
                        "Company Closing Conditions – Concise": summarizer.summarize_company_conditions,
                        "Special Conditions Summary": summarizer.summarize_additional_key_conditions,
                        "Non-Customary Conditions": summarizer.summarize_non_customary_conditions,
                    }

                    for label, fn in task_map.items():
                        clause_type = "concise" if "Concise" in label else "fullsome"
                        future = executor.submit(fn)
                        futures_meta.append(
                            (future, section_obj, category, clause_type, label))

                else:
                    for base in self.active_clauses[category]:
                        future = executor.submit(
                            summarize_clause, category, base)
                        # 'bulk' for merged later
                        futures_meta.append(
                            (future, section_obj, category, "bulk", None))

            future_to_meta = {fut: (section_obj, category, clause_type, label)
                              for fut, section_obj, category, clause_type, label in futures_meta}

            # Collect and assign results
            for future in as_completed(future_to_meta.keys(), timeout=None):
                section_obj, category, clause_type, label = future_to_meta[future]
                result = future.result()

                if clause_type == "bulk":
                    section_obj[category]["concise"].update(result["concise"])
                    section_obj[category]["fullsome"].update(
                        result["fullsome"])
                else:
                    section_obj[category][clause_type][label] = result

        return sections
