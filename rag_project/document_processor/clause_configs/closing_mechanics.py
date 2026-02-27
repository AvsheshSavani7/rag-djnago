CLOSING_MECHANICS_CLAUSE_CONFIG = {
    "Target Date - Concise": {
        # ✅ Top-level key in JSON
        "section": "closing_mechanics",
        "subsection": "target_date",             # ✅ The nested list inside

        "summary_field": "target_date",          # ✅ Matches field_name
        "reference_field": "reference_section",               # ✅ Field inside the item
        "output_source_field": "clause_text",                 # ✅ Source for LLM input

        "prompt_template": (
            "You are a legal analyst summarizing the target date in a merger agreement.\n"
            "{}\n"
            "Write a professional, clear, one-sentence bullet summarizing the target date.\n"
            "Bullet:"
        ),

        "if_false_summary": "No target date is included in the agreement.",
        "if_missing_summary": "Target date summary is missing.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "bullet",
        "max_words": 50,
        "summary_type": "concise",
    },

    "Marketing Period - Fulsome": {
        "section": "closing_mechanics",
        "subsection": "marketing_period",
        "included_field": "has_marketing_period",
        "summary_field": "marketing_period_details",
        "reference_field": "reference_section",
        "output_source_field": "clause_text",

        "prompt_template": (
            "You are a legal analyst reviewing a merger agreement. Below is the full text of a clause:\n\n"
            "{}\n\n"
            "Your task is to write a precise, paraphrased 1-paragraph summary of the clause that:\n"
            "- States the main obligation (e.g., use of commercially reasonable efforts to operate in ordinary course)\n"
            "- Clearly lists the exceptions or carve-outs (e.g., required by law, agreed in writing, etc.)\n"
            "- Includes any specific operational duties or examples mentioned (e.g., preserving relationships, asset protection)\n"
            "- Uses some key quoted phrases from the clause if helpful, but avoid copying large sections\n"
            "- Avoids generic phrasing — aim to capture the nuance of the original clause.\n\n"
            "Write only one paragraph."
        ),

        "if_false_summary": "No marketing period is included in the agreement.",
        "if_missing_summary": "Marketing period summary is missing.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "paragraph",
        "max_words": 150,
        "summary_type": "fulsome",
    },

    "Inside Date - Fulsome": {
        "section": "closing_mechanics",
        "subsection": "inside_date",
        "included_field": "has_inside_date",
        "summary_field": "inside_date_details",
        "reference_field": "reference_section",
        "output_source_field": "clause_text",

        "prompt_template": (
            "You are a legal analyst reviewing a merger agreement. Below is the full text of a clause:\n\n"
            "{}\n\n"
            "Your task is to write a precise, paraphrased 1-paragraph summary of the clause that:\n"
            "- States the main obligation (e.g., use of commercially reasonable efforts to operate in ordinary course)\n"
            "- Clearly lists the exceptions or carve-outs (e.g., required by law, agreed in writing, etc.)\n"
            "- Includes any specific operational duties or examples mentioned (e.g., preserving relationships, asset protection)\n"
            "- Uses some key quoted phrases from the clause if helpful, but avoid copying large sections\n"
            "- Avoids generic phrasing — aim to capture the nuance of the original clause.\n\n"
            "Write only one paragraph."
        ),

        "if_false_summary": "No inside date is included in the agreement.",
        "if_missing_summary": "Inside date summary is missing.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "paragraph",
        "max_words": 150,
        "summary_type": "fulsome",

    }
}
