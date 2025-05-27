import os
import json
from dotenv import load_dotenv
import openai


class LLMClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.client = openai.OpenAI()

    def generate_bullet(self, prompt, temperature=0.3, max_tokens=160):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a legal analyst summarizing merger agreement conditions using concise M&A terminology."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return "" + response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM error: {e}"


class ConditionSummarizer:
    def __init__(self, data):
        self.data = data
        self.llm = LLMClient()

    def get_answer(self, field):
        return next((item.get("answer") for item in self.data if item.get("field_name") == field), None)

    def get_bool(self, field):
        return any(item.get("field_name") == field and item.get("answer") is True for item in self.data)

    def summarize_mutual_conditions(self):
        conditions = self.get_answer("mutual_conditions_list")
        if not conditions:
            return "No mutual conditions found."
        prompt = (
            "Below are mutual closing conditions from a merger agreement:\n\n"
            + "\n".join(conditions) + "\n\n"
            "Summarize as a single sentence: 'Mutual conditions include: ...' Use M&A terms."
        )
        return self.llm.generate_bullet(prompt)

    def summarize_parent_conditions(self):
        conditions = self.get_answer("parent_closing_conditions")
        if not conditions:
            return " No Parent closing conditions found."
        prompt = (
            "Conditions to Parent’s obligation to close:\n\n"
            + "\n".join(conditions) + "\n\n"
            "Summarize in one sentence using M&A terms."
        )
        return self.llm.generate_bullet(prompt)

    def summarize_company_conditions(self):
        conditions = self.get_answer("company_closing_conditions")
        if not conditions:
            return " No Company closing conditions found."
        prompt = (
            "Conditions to Company’s obligation to close:\n\n"
            + "\n".join(conditions) + "\n\n"
            "Summarize in one sentence using M&A terms."
        )
        return self.llm.generate_bullet(prompt)

    def summarize_additional_key_conditions(self):
        items = []
        seen_keywords = set()

        def include_unique(label, keywords):
            if not any(kw.lower() in seen_keywords for kw in keywords):
                items.append(label)
                seen_keywords.update(keywords)

        if self.get_bool("target_shareholder_approval_required"):
            threshold = self.get_answer(
                "target_shareholder_approval_threshold")
            include_unique(f"Shareholder approval (majority threshold: {threshold})" if threshold
                           else "Shareholder approval required", ["shareholder", "approval"])

        if self.get_bool("hsr_clearance_required"):
            include_unique("HSR Clearance", ["hsr"])

        customary = self.get_answer("other_customary_conditions")
        if isinstance(customary, list):
            for line in customary:
                line_clean = line.strip()
                if line_clean and line_clean.lower() != "not found" and \
                        not any(kw in line_clean.lower() for kw in seen_keywords):
                    items.append(line_clean)

        if not items:
            return " No additional key conditions specified."

        prompt = (
            "Summarize the following list into a single sentence starting with:\n"
            "'Key conditions include: ...'\n\n"
            + "\n".join(f"- {x}" for x in items)
        )
        return self.llm.generate_bullet(prompt)

    def summarize_non_customary_conditions(self):
        conditions_map = {
            "financial_conditions_minimum_cash_threshold": "Minimum cash threshold",
            "financial_conditions_maximum_net_debt": "Maximum net debt",
            "financial_conditions_minimum_any_condition": "Minimum financial metric",
            "tax_conditions_tax_opinion_required": "Tax opinion condition",
        }

        output = []
        for field, label in conditions_map.items():
            answer = self.get_answer(field)
            if isinstance(answer, str) and answer.strip().lower() != "not found":
                output.append(f" Non-customary condition: {label}")
            elif isinstance(answer, list) and any(isinstance(line, str) and line.strip().lower() != "not found" for line in answer):
                output.append(f" Non-customary condition: {label}")
            elif isinstance(answer, (int, float)) or answer is True:
                output.append(f" Non-customary condition: {label}")

        return "\n".join(output) if output else "(No non-customary conditions identified.)"
