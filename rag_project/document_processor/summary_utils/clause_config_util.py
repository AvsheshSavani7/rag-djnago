from ..clause_configs.clause_config_ordinary_course import ORDINARY_COURSE_CLAUSE_CONFIG
from ..clause_configs.clause_config_non_solicitation import (
    NON_SOLICITATION_CLAUSE_CONFIG,
)
from ..clause_configs.clause_config_conditions_to_closing import (
    CLOSING_CONDITIONS_CLAUSE_CONFIG,
)
from ..clause_configs.financing_summary import (
    FINANCING_SUMMARY_CLAUSE_CONFIG)
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

        def summarize_clause(category, base):
            concise_label = f"{base} - Concise"
            fulsome_label = f"{base} - Fulsome"
            result = {"category": category, "concise": {}, "fullsome": {}}

            logger.info(f"DEBUG: concise_label → {concise_label}")
            logger.info(f"DEBUG: fulsome_label → {fulsome_label}")

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
            futures = []

            for category in self.active_clauses:
                section_obj = {category: {"concise": {}, "fullsome": {}}}
                sections.append(section_obj)

                if category == "best_efforts":
                    summarizer = RegulatorySummarizer(schema.get(
                        "best_efforts"), parent_name="James Hardie")

                    # Create futures for all best_efforts operations instead of running sequentially
                    concise_futures = [
                        executor.submit(
                            summarizer.summarize_divestiture_section),
                        executor.submit(
                            summarizer.summarize_litigation_commitments)
                    ]

                    fullsome_futures = [
                        executor.submit(summarizer.summarize_hsr_filing),
                        executor.submit(summarizer.summarize_cfiius_filing),
                        executor.submit(summarizer.summarize_foreign_filing),
                        executor.submit(summarizer.summarize_effort_standard),
                        executor.submit(
                            summarizer.summarize_withdrawal_controls),
                        executor.submit(summarizer.summarize_timing_agreement),
                        executor.submit(
                            summarizer.summarize_divestiture_cap_fulsome),
                        executor.submit(
                            summarizer.summarize_prior_approval_commitment),
                        executor.submit(
                            summarizer.summarize_transaction_interference),
                        executor.submit(
                            summarizer.summarize_second_request_certification),
                        executor.submit(
                            summarizer.summarize_ftc_warning_letter_handling)
                    ]

                    # Add results as they complete
                    section_obj[category]["concise"]["Divestiture – Concise"] = concise_futures[0].result(
                    )
                    section_obj[category]["concise"]["Litigation – Concise"] = concise_futures[1].result(
                    )

                    section_obj[category]["fullsome"]["HSR Filing – fullsome"] = fullsome_futures[0].result(
                    )
                    section_obj[category]["fullsome"]["CFIUS Filing – fullsome"] = fullsome_futures[1].result(
                    )
                    section_obj[category]["fullsome"]["Foreign Filing – fullsome"] = fullsome_futures[2].result(
                    )
                    section_obj[category]["fullsome"]["Effort Standard – fullsome"] = fullsome_futures[3].result(
                    )
                    section_obj[category]["fullsome"]["Withdrawal Control – fullsome"] = fullsome_futures[4].result(
                    )
                    section_obj[category]["fullsome"]["Timing Agreement Restriction – fullsome"] = fullsome_futures[5].result(
                    )
                    section_obj[category]["fullsome"]["Divestiture Cap Fulfsome"] = fullsome_futures[6].result(
                    )
                    section_obj[category]["fullsome"]["Prior Approval Commitment – fullsome"] = fullsome_futures[7].result(
                    )
                    section_obj[category]["fullsome"]["Transaction Interference – fullsome"] = fullsome_futures[8].result(
                    )
                    section_obj[category]["fullsome"]["Second Request Certification – fullsome"] = fullsome_futures[9].result(
                    )
                    section_obj[category]["fullsome"]["FTC Warning Letter Handling – fullsome"] = fullsome_futures[10].result(
                    )

                elif category == "board_approvals":
                    summarizer = BoardApprovalsSummarizer(
                        schema.get("board_approval"))
                    concise_futures = [executor.submit(
                        summarizer.get_consice_summary)]

                    fullsome_futures = [executor.submit(
                        summarizer.get_fulsome_summary)]

                    section_obj[category]["concise"]["Board Approvals – Concise"] = concise_futures[0].result(
                    )

                    if concise_futures.__len__() > 1:
                        section_obj[category]["concise"]["Board Approvals – Concise2"] = concise_futures[1].result(
                        )

                    section_obj[category]["fullsome"]["Board Approvals – fullsome"] = fullsome_futures[0].result(
                    )

                elif category == "conditions_to_closing":
                    summarizer = ConditionSummarizer(
                        schema.get("conditions_to_closing"))

                    # Create futures for all conditions_to_closing operations
                    concise_futures = [
                        executor.submit(
                            summarizer.summarize_mutual_conditions),
                        executor.submit(
                            summarizer.summarize_parent_conditions),
                        executor.submit(
                            summarizer.summarize_company_conditions)
                    ]

                    fullsome_futures = [
                        executor.submit(
                            summarizer.summarize_additional_key_conditions),
                        executor.submit(
                            summarizer.summarize_non_customary_conditions)
                    ]

                    # Add results as they complete
                    section_obj[category]["concise"]["Mutual Conditions – Concise"] = concise_futures[0].result(
                    )
                    section_obj[category]["concise"]["Parent Closing Conditions – Concise"] = concise_futures[1].result(
                    )
                    section_obj[category]["concise"]["Company Closing Conditions – Concise"] = concise_futures[2].result(
                    )
                    section_obj[category]["fullsome"]["Special Conditions Summary"] = fullsome_futures[0].result(
                    )
                    section_obj[category]["fullsome"]["Non-Customary Conditions"] = fullsome_futures[1].result()

                else:
                    for base in self.active_clauses[category]:
                        futures.append(executor.submit(
                            summarize_clause, category, base))

            # Merge threaded results
            for future in as_completed(futures):
                result = future.result()
                for sec in sections:
                    if result["category"] in sec:
                        sec[result["category"]]["concise"].update(
                            result["concise"])
                        sec[result["category"]]["fullsome"].update(
                            result["fullsome"])

        return sections
