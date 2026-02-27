import os
import json
from dotenv import load_dotenv
import openai


class LLMClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        openai.api_key = self.api_key
        self.client = openai.OpenAI()

    def summarize(self, prompt, temperature=0.3, max_tokens=120):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a legal analyst summarizing merger agreement terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return "" + response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM error: {e}"


class RegulatorySummarizer:
    def __init__(self, data,  parent_name="the parent"):
        self.data = data
        self.llm = LLMClient()
        self.parent_name = parent_name

    def get_fields(self, section_name):
        return {item["field_name"]: item["answer"] for item in self.data.get(section_name, []) if isinstance(item, dict)}

    def generate_bullet(self, prompt):
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a legal analyst summarizing merger agreement terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            text = response.choices[0].message.content.strip()
            if not text:
                print("⚠️ LLM returned an empty string. Falling back to manual summary.")
                text = "Divestiture cap obligations not clearly specified."
            return "" + text
        except Exception as e:
            print(f"LLM error: {e}")
            return None

    def summarize_divestiture_section(self, use_llm=True):
        fields = {item["field_name"]: item["answer"]
                  for item in self.data.get("divestiture_commitments", [])}
        required = fields.get("divestiture_required")
        if required is False or (required is None and not any(k for k in fields if k.startswith("divestiture_cap"))):
            return "No divestiture commitments are required under the terms of the agreement."

        bullets = []

        target_prompt = (
            "You are a legal and finance analyst summarizing M&A contract terms. Write a single concise bullet summarizing the divestiture cap for the target based on the data below. Use professional tone and precise legal language.\n"
            # f"Cap Value: {fields.get('divestiture_cap_target_value')}\n"
            # f"Unit: {fields.get('divestiture_cap_target_unit')}\n"
            # f"Basis: {fields.get('divestiture_cap_target_basis')}\n"
            # f"Scope: {fields.get('divestiture_cap_target_basis_scope')}\n"
            f"Notes: {fields.get('divestiture_cap_target_notes')}\n"

        )

        buyer_prompt = (
            "You are a legal and finance analyst summarizing M&A contract terms for a knowledgeable audience."
            "Write one single **concise bullet point**, no more than **25 words**, that explains the divestiture cap as it applies to the buyer and the buyer's businesses. "
            "Use precise legal language but avoid verbosity. If the buyer is not obligated to take any remedy actions on its business, clearly state that. "
            "If the cap is zero or not applicable, explicitly note it.\n\n"
            # f"Cap Value: {fields.get('divestiture_cap_buyer_value')}\n"
            # f"Unit: {fields.get('divestiture_cap_buyer_unit')}\n"
            # f"Basis: {fields.get('divestiture_cap_buyer_basis')}\n"
            # f"Scope: {fields.get('divestiture_cap_buyer_basis_scope')}\n"
            f"Notes: {fields.get('divestiture_cap_buyer_notes')}\n"

        )

        target_bullet = self.generate_bullet(
            target_prompt) if use_llm else f"Target business units generating more than ${fields.get('divestiture_cap_target_value')}M in {fields.get('divestiture_cap_target_basis')} are subject to a divestiture cap."
        buyer_bullet = self.generate_bullet(
            buyer_prompt) if use_llm else f"Buyer is not required to divest any of its own business units. {fields.get('divestiture_cap_buyer_notes') or ''}"

        if target_bullet:
            bullets.append(target_bullet)
        if buyer_bullet:
            bullets.append(buyer_bullet)

        return "\n".join(bullets)

    def generate_llm_summary_filing(self, jurisdiction, deadline, notes, default_clause, use_llm=True):
        if not use_llm:
            if deadline and isinstance(deadline, int):
                return f"Parties shall file under {jurisdiction} no later than {deadline} Business Days from the date of this Agreement.\n  {default_clause}"
            elif notes:
                return f"{notes.strip()}\n  {default_clause}"
            else:
                return f"{jurisdiction} filing is required.\n   {default_clause}"

        prompt = (
            f"Summarize the filing obligation for {jurisdiction} in one bullet point.\n"
            f"Deadline (if any): {deadline}\n"
            f"Notes: {notes}\n"
            "Avoid redundancy. If both deadline and notes exist, combine them logically. Be concise and formal.\n"

        )
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a legal analyst summarizing merger agreement terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=120
            )
            return "" + response.choices[0].message.content.strip() + "\n   " + str(default_clause)
        except Exception as e:
            print(f"LLM error: {e}")
            return None

    def summarize_litigation_commitments(self, parent_name="the parent"):
        fields = {item["field_name"]: item["answer"]
                  for item in self.data.get("litigation_commitments", [])}
        status = fields.get("litigation_requirement_status")
        if status == "Explicit Requirement to Litigate":
            return f"{parent_name} is required to litigate in defense of the transaction."
        elif status == "No Requirement to Litigate":
            return "The agreement does not impose any obligation to litigate in defense of the transaction."
        return ""

    def summarize_hsr_filing(self, use_llm=True):
        fields = {item["field_name"]: item["answer"]
                  for item in self.data.get("regulatory_fillings_hsr", [])}
        if fields.get("hsr_required") is not True:
            return "HSR filing is not required under the agreement."
        deadline = fields.get("hsr_filing_deadline_days")
        notes = fields.get("hsr_notes")
        reference = fields.get(
            "hsr_filing_deadline_clause_reference", "Section 5.4(a)")
        return self.generate_llm_summary_filing("HSR", deadline, notes, reference, use_llm)

    def summarize_cfiius_filing(self, use_llm=True):
        fields = {item["field_name"]: item["answer"]
                  for item in self.data.get("regulatory_fillings_cfiius", [])}
        required = fields.get("cfiius_required")
        if required is not True:
            return "CFIUS filing is not required under the agreement. (not included in final output)"
        deadline_days = fields.get("cfiius_filing_deadline_days")
        notes = fields.get("cfiius_notes")
        reference = fields.get(
            "cfiius_filing_deadline_clause_reference", "Section 5.4")
        return self.generate_llm_summary_filing("CFIUS", deadline_days, notes, reference, use_llm)

    def summarize_foreign_filing(self, use_llm=True):
        fields = {item["field_name"]: item["answer"]
                  for item in self.data.get("regulatory_fillings_foreign", [])}
        required = fields.get("foreign_required")
        if required is not True:
            return "Foreign regulatory filings are not required under the agreement. (not included in final output)"
        deadline_days = fields.get("foreign_filing_deadline_days")
        notes = fields.get("foreign_notes")
        reference = fields.get(
            "foreign_filing_deadline_clause_reference", "Section 5.4")
        return self.generate_llm_summary_filing("Foreign", deadline_days, notes, reference, use_llm)

    def summarize_effort_standard(self, use_llm=True):
        regulatory_efforts = self.data.get("regulatory_efforts", [])
        fields = {
            item["field_name"]: item["answer"]
            for item in regulatory_efforts
            if isinstance(item, dict) and item.get("field_name") == "effort_standard"
        }
        standard = fields.get("effort_standard")

        if not standard:
            return ""

        if not use_llm:
            return f"Parties shall use {standard} efforts to receive all required regulatory approvals."

        prompt = (
            "You are a legal analyst summarizing M&A agreement terms.\n"
            "Write one concise bullet point describing the regulatory effort standard used by the parties.\n"
            f"Effort Standard: {standard}\n"
            "Use professional legal tone. If applicable, clarify that the standard applies to obtaining regulatory approvals.\n"

        )

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a legal analyst summarizing merger agreement terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            text = response.choices[0].message.content.strip()
            if not text:
                text = f"Parties shall use {standard} efforts to receive all required regulatory approvals."
            return "" + text
        except Exception as e:
            print(f"LLM error: {e}")
            return f"Parties shall use {standard} efforts to receive all required regulatory approvals."

    def summarize_withdrawal_controls(self, use_llm=True):
        fields = {
            item["field_name"]: item["answer"]
            for item in self.data.get("withdrawal_and_timing_controls_withdrawal", [])
            if isinstance(item, dict) and item.get("field_name") in ("withdrawal_control_type", "withdrawal_control_type_notes")
        }
        control_type = fields.get("withdrawal_control_type")
        notes = fields.get("withdrawal_control_type_notes")

        if not control_type:
            return ""

        if not use_llm:
            return f"Consent requirement: {control_type}. {notes or ''}"

        prompt = (
            "You are a legal analyst summarizing M&A agreement terms.\n"
            "Write one concise bullet point describing the type of consent required for withdrawal of regulatory filings and any conditions attached.\n"
            f"Control Type: {control_type}\n"
            f"Notes: {notes or 'None'}\n"
            "Use professional legal tone. Avoid redundancy.\n"

        )

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a legal analyst summarizing merger agreement terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            return "" + response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM error: {e}")
            return f"Consent requirement: {control_type}. {notes or ''}"

    def summarize_timing_agreement(self, use_llm=True):
        fields = {
            item["field_name"]: item["answer"]
            for item in self.data.get("withdrawal_and_timing_controls_timing", [])
            if isinstance(item, dict) and item.get("field_name") in ("timing_agreement_restriction", "timing_agreement_restriction_notes")
        }
        restriction = fields.get("timing_agreement_restriction")
        notes = fields.get("timing_agreement_restriction_notes")

        if restriction is not True and not notes:
            return ""

        if not use_llm:
            return f"Timing agreement restriction: {notes or 'Not specified.'}"

        prompt = (
            "You are a legal analyst summarizing M&A agreement terms.\n"
            "Write one concise bullet point, with no more than 30 words, explaining any restriction on entering timing agreements related to regulatory filings.\n"
            "Ensure the response focuses on restrictions and and consents.\n"
            f"Restriction Present: {restriction}\n"
            f"Notes: {notes or 'None'}\n"
            "Use formal legal tone. Avoid redundancy. Bullet:"
        )

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a legal analyst summarizing merger agreement terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            return "" + response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM error: {e}")
            return f"Timing agreement restriction: {notes or 'Not specified.'}"

    def summarize_divestiture_cap_fulsome(self, use_llm=True):
        fields = {
            item["field_name"]: item["answer"]
            for item in self.data.get("divestiture_commitments", [])
            if isinstance(item, dict)
        }

        if fields.get("divestiture_required") is False and not any(k.startswith("divestiture_cap") for k in fields):
            return "No divestiture commitments apply under the agreement."

        if not use_llm:
            return "Divestiture cap provisions are included, but fulsome summary not generated (LLM disabled)."

        prompt = (
            "You are a legal analyst summarizing M&A agreement terms.\n"
            "Write a comprehensive summary, not to exceed 250 tokens, for the divestiture cap commitments in the agreement.\n"
            "Clearly distinguish between divestiture cap limitations for the Target vs. the Buyer (Parent).\n"
            "For the Target, explain what types of business units are exempt from remedy obligations due to size thresholds or revenue.\n"
            "For the Buyer, state clearly if the buyer (Parent and its Subsidiaries) is not required to take any remedy actions.\n"
            f"Target Cap Notes: {fields.get('divestiture_cap_target_notes') or 'None'}\n"
            f"Buyer Cap Notes: {fields.get('divestiture_cap_buyer_notes') or 'None'}\n"
            "Avoid duplicating contract language. Use a formal legal tone. Write as one complete paragraph. Be concise, and do not state obvious items.\n"
            "End your response cleanly. Avoid trailing phrases that suggest an unfinished conclusion."
            "Summary:"
        )

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a legal analyst summarizing merger agreement terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            return "" + response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM error: {e}")
            return "Fulsome divestiture cap summary could not be generated due to LLM error."

    def summarize_prior_approval_commitment(self, use_llm=True):
        if not isinstance(self.data, dict):
            print(
                "⚠️ Skipping prior approval summary — expected a dict but received a list.")
            return ""

        fields = {
            item["field_name"]: item["answer"]
            for item in self.data.get("prior_approval_commitment", [])
            if isinstance(item, dict) and item.get("field_name") in ("addressed", "obligations_summary", "clause_reference")
        }

        addressed = fields.get("addressed")
        summary = fields.get("obligations_summary")
        reference = fields.get("clause_reference")

        if addressed is not True and not summary:
            return ""

        if not use_llm:
            ref_line = f"\n   {reference}" if reference and reference.lower(
            ) != "not found" else ""
            return f"Prior approval commitments: {summary or 'Summary not provided.'}{ref_line}"

        prompt = (
            "You are a legal analyst summarizing M&A agreement terms.\n"
            "Write one concise bullet point summarizing any prior approval commitments under the agreement.\n"
            f"Addressed: {addressed}\n"
            f"Summary: {summary or 'None'}\n"
            f"Clause Reference: {reference or 'N/A'}\n"
            "Use professional legal tone. Be concise and clear.\n"

        )

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a legal analyst summarizing merger agreement terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=120
            )
            bullet = response.choices[0].message.content.strip()
            ref_line = f"\n   {reference}" if reference and reference.lower(
            ) != "not found" else ""
            return "" + bullet + ref_line
        except Exception as e:
            print(f"LLM error: {e}")
            ref_line = f"\n   {reference}" if reference and reference.lower(
            ) != "not found" else ""
            return f"Prior approval commitments: {summary or 'Summary not provided.'}{ref_line}"

    def summarize_transaction_interference(self, use_llm=True):
        fields = {
            item["field_name"]: item["answer"]
            for item in self.data.get("transaction_interference", [])
            if isinstance(item, dict) and item.get("field_name") in (
                "restriction_on_other_transactions",
                "interference_clause_text",
                "interference_clause_text_clause_reference"
            )
        }

        restriction = fields.get("restriction_on_other_transactions")
        notes = fields.get("interference_clause_text")
        reference = fields.get("interference_clause_text_clause_reference")

        if not restriction and not notes:
            return ""

        if not use_llm:
            ref_line = f"\n   {reference}" if reference and reference.lower(
            ) != "not found" else ""
            return f"Interference restriction: {notes or 'Not specified.'}{ref_line}"

        prompt = (
            "You are a legal analyst summarizing M&A agreement terms.\n"
            "Write one concise bullet point explaining any restrictions on alternative transactions that could interfere with the agreed merger.\n"
            f"Restricted: {restriction}\n"
            f"Notes: {notes or 'None'}\n"
            f"Clause Reference: {reference or 'N/A'}\n"
            "Use precise legal language and a formal tone. Be clear and avoid duplication. Do not cite the section number or clause in the sentence — it will be added separately.\n"
            "Start directly with the parties or actions involved.\n"

        )

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a legal analyst summarizing merger agreement terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=120
            )
            bullet = response.choices[0].message.content.strip()
            ref_line = f"\n   {reference}" if reference and reference.lower(
            ) != "not found" else ""
            return "" + bullet + ref_line
        except Exception as e:
            print(f"LLM error: {e}")
            ref_line = f"\n   {reference}" if reference and reference.lower(
            ) != "not found" else ""
            return f"Interference restriction: {notes or 'Summary not provided.'}{ref_line}"

    def summarize_second_request_certification(self, use_llm=True):
        fields = {
            item["field_name"]: item["answer"]
            for item in self.data.get("second_request_certification", [])
            if isinstance(item, dict) and item.get("field_name") in (
                "deadline_specified",
                "certification_timeline_notes",
                "certification_responsibility_party"
            )
        }

        deadline = fields.get("deadline_specified")
        notes = fields.get("certification_timeline_notes")
        party = fields.get("certification_responsibility_party")

        if deadline is False:
            return "There is no contractually specified deadline by which the parties shall certify compliance with any second request for information under the HSR Act."

        if not deadline and not notes:
            return ""

        if not use_llm:
            return f"Certification timing: {notes or 'Summary not provided.'}"

        prompt = (
            "You are a legal analyst summarizing M&A agreement terms.\n"
            "Write one concise bullet point explaining the timing and party responsibility for certifying compliance with a second request under the HSR Act.\n"
            f"Deadline Specified: {deadline}\n"
            f"Timeline Notes: {notes or 'None'}\n"
            f"Responsible Party: {party or 'Not specified'}\n"
            "Do not include clause references. Use formal legal tone. Bullet:"
        )

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a legal analyst summarizing merger agreement terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=120
            )
            return "" + response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM error: {e}")
            return f"Certification timing: {notes or 'Summary not provided.'}"

    def summarize_ftc_warning_letter_handling(self, use_llm=True):
        fields = {
            item["field_name"]: item["answer"]
            for item in self.data.get("ftc_warning_letter_handling", [])
            if isinstance(item, dict) and item.get("field_name") in (
                "addressed",
                "handling_notes",
                "effect_on_closing_condition",
                "addressed_notes"
            )
        }

        addressed = fields.get("addressed")
        notes = fields.get("handling_notes")
        closing_impact = fields.get("effect_on_closing_condition")
        extra = fields.get("addressed_notes")

        # ✅ Handles both actual False and string "false"
        if str(addressed).lower() == "false":
            return "There is no consideration per the contract given to the receipt of an FTC warning letter."

        if not addressed and not notes:
            return ""

        if not use_llm:
            return f"FTC warning letter handling: {notes or 'Summary not provided.'}"

        prompt = (
            "You are a legal analyst summarizing M&A agreement terms.\n"
            "Write one concise bullet point explaining how the agreement addresses the receipt of an FTC warning letter.\n"
            "If the letter is addressed, summarize the procedure and clarify whether it affects closing conditions.\n"
            f"Addressed: {addressed}\n"
            f"Handling Notes: {notes or 'None'}\n"
            f"Effect on Closing: {closing_impact or 'Unknown'}\n"
            f"Additional Notes: {extra or 'None'}\n"
            "Do not include clause references. Write clearly and formally in one sentence.\n"

        )

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a legal analyst summarizing merger agreement terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=120
            )
            return "" + response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM error: {e}")
            return f"FTC warning letter handling: {notes or 'Summary not provided.'}"

    # Other summarization methods (hsr, cfiius, etc.) would be structured similarly
