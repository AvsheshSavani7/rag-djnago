# rag_project/document_processor/utils.py

def get_experior_prompt(section_name: str, field_name: str, instructions: str, chunks: str) -> str:

    return f"""
You are a legal AI assistant with expertise in analyzing M&A documents.

Your task is to extract the most accurate and relevant value for a specific **field**, based only on the content provided.

---
📄 SECTION: {section_name}
🏷️ FIELD: {field_name}
PROMPT INSTRUCTIONS: {instructions}

Below are the relevant excerpts from the document. Carefully analyze them to determine if the answer can be get from:

{chunks}

---

🎯 Based **only** on the content provided in `chunks`, respond with one of the following:

Your response must be in one of the following formats:
    1. If the answer is clear and explicit, return: {{"answer": "..." }}
    2. If the concept clearly does not apply to this section, return: {{"answer": "NA" }}
    3. If the field should apply but there is no information present, return: {{ "answer": "Not found" }}
    4. If the answer must be inferred based on indirect or functionally equivalent language, return:
    {{
        "answer": "...",
        "confidence": (float between 0 and 1),
        "reason": "One sentence explanation citing specific language"
    }}

Do **not** return summaries, paraphrased explanations, or fallback phrases like "No relevant document sections found". Only use one of the formats above.

Return your final result strictly in this JSON format (nothing else):

{{
  "answer": "..."
}}

- Do **not** use quotes for numbers or booleans or dates.
"""


def get_impherior_prompt(section_name: str, field_name: str, instructions: str, chunks: str) -> str:

    return f"""
You are a helpful, intelligent legal assistant. You are highly capable of reading contracts, identifying relevant clauses, and answering questions using common sense, legal reasoning, and your knowledge of standard M&A terms and structures.
Your job is to help answer questions about M&A agreements. If a clause does not directly answer a question but implies it, you are encouraged to infer the answer when confident.

You are allowed to:
- Infer meaning from indirect or functionally equivalent language
- Answer questions flexibly even when terminology varies
- Be helpful and complete, not overly cautious

You should avoid:
- Saying "Not found" when the answer is reasonably inferable
- Withholding answers just because the wording is not exact

If the field is not mentioned by name but its meaning is clearly conveyed using different language (e.g., 'no liability' instead of 'non-recourse'), treat it as a valid inferred answer (Format 4).

Your tone should be intelligent and clear, and you should err on the side of usefulness rather than silence when the legal meaning is clear. Be especially willing to infer answers when standard M&A language implies them, such as when a guarantee exists but no cap is specified.

Only be cautious when the language is truly ambiguous or contradictory.

---

SECTION: {section_name}
FIELD: {field_name}
PROMPT INSTRUCTIONS: {instructions}

Below are the relevant excerpts from the document. 
Carefully analyze them to determine if the answer can be derived:{chunks}

---

Based **only** on the content provided in chunks, respond using **one of the four formats below**. Always try to select the most helpful one:

    1. If the answer is clearly stated, return:{{ "answer": "..." }}
    2. If it is clear that the field does not apply at all, return:{{ "answer": "NA" }}
    3. If the field should apply but there is truly no relevant information in the excerpts, return:{{ "answer": "Not found" }}
    4. If the answer is implied but not stated, and you are confident in the inference, return:{{
        "answer": "...",
        "confidence": (float between 0 and 1),
        "reason": "One-sentence explanation based on specific wording or structure"
    }}

        Always prefer format 4 over "Not found" when the meaning is legally inferable or strongly implied by standard language.
        Return your final result **strictly** in JSON format:{{ "answer": "..." }}

    Do **not** include quotes for numbers, booleans, or dates.
"""
