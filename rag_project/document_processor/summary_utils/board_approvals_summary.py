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


class BoardApprovalsSummarizer:
    def __init__(self, data):
        self.data = data
        self.llm = LLMClient()
        print(f"DEBUG: BoardApprovalsSummarizer data → {self.data}")

    def get_answer(self, field):
        return next((item.get("answer") for item in self.data if item.get("field_name") == field), None)

    def get_bool(self, answer):
        return True if (str(answer).lower() == "true") else False if (str(answer).lower() == "false") else answer

    def get_fields(self, section_name):
        return {item["field_name"]: item["answer"] for item in self.data.get(section_name, []) if isinstance(item, dict)}

    def get_clause_texts(self, section_name):
        return {item["field_name"]: item["clause_text"] for item in self.data.get(section_name, []) if isinstance(item, dict)}

    def get_fulsome_summary(self):
        bullet_point = self.get_consice_summary()

        fields = self.get_fields("board_approval")
        clause_texts = self.get_clause_texts("board_approval")
        # clause_text
        target_board_approval = {"target_board_approval": self.get_bool(
            fields.get("target_board_approval")), "clause_text": clause_texts.get("target_board_approval")}
        target_board_unanimous = {"target_board_unanimous": self.get_bool(
            fields.get("target_board_unanimous")), "clause_text": clause_texts.get("target_board_unanimous")}
        acquirer_board_approval = {"acquirer_board_approval": self.get_bool(
            fields.get("acquirer_board_approval")), "clause_text": clause_texts.get("acquirer_board_approval")}
        acquirer_board_unanimous = {"acquirer_board_unanimous": self.get_bool(
            fields.get("acquirer_board_unanimous")), "clause_text": clause_texts.get("acquirer_board_unanimous")}
        fields_witth_text = [target_board_approval, target_board_unanimous,
                             acquirer_board_approval, acquirer_board_unanimous]

        prompt = (
            "Below are board approval conditions from a merger agreement:\n\n"
            + f"{fields_witth_text}" + "\n\n"
            "Summarize as a single paragraph. Use M&A terms."
        )
        return self.llm.generate_bullet(prompt)

    def get_consice_summary(self):
        fields = self.get_fields("board_approval")
        target_board_approval = self.get_bool(
            fields.get("target_board_approval"))
        target_board_unanimous = self.get_bool(
            fields.get("target_board_unanimous"))
        acquirer_board_approval = self.get_bool(
            fields.get("acquirer_board_approval"))
        acquirer_board_unanimous = self.get_bool(
            fields.get("acquirer_board_unanimous"))

        bullet_point = []

        if target_board_approval and target_board_unanimous and acquirer_board_approval and acquirer_board_unanimous:
            bullet_point.append(
                "Merger has been unanimously approved by the boards of the Company and the Parent.")
        else:
            if target_board_approval and target_board_unanimous:
                bullet_point.append(
                    "Merger has been approved by the board of the Company unanimously.")
            elif target_board_approval and not target_board_unanimous:
                bullet_point.append(
                    "Merger has been approved by the board of the Company, but not unanimously.")
            elif not target_board_approval and not target_board_unanimous:
                bullet_point.append(
                    "Merger has not been approved by the board of the Company.")
            else:
                bullet_point.append(
                    "Board approval not found for the Company.")

            if acquirer_board_approval and acquirer_board_unanimous:
                bullet_point.append(
                    "Merger has been approved by the board of the Parent unanimously.")
            elif acquirer_board_approval and not acquirer_board_unanimous:
                bullet_point.append(
                    "Merger has been approved by the board of the Parent, but not unanimously.")
            elif not acquirer_board_approval and not acquirer_board_unanimous:
                bullet_point.append(
                    "Merger has not been approved by the board of the Parent.")
            else:
                bullet_point.append(
                    "Board approval not found for the Parent.")

        return "\n".join(bullet_point)
