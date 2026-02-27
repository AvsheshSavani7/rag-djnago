ORDINARY_COURSE_CLAUSE_CONFIG = {
    "Ordinary Course - Concise": {
        "section": "ordinary_course",                         # ✅ Top-level key in JSON
        "subsection": "ordinary_course_covenant",             # ✅ The nested list inside

        "summary_field": "concise_standard_summary",          # ✅ Matches field_name
        "reference_field": "reference_section",               # ✅ Field inside the item
        "output_source_field": "clause_text",                 # ✅ Source for LLM input

        "prompt_template": (
            "You are a legal analyst summarizing the ordinary course covenant in a merger agreement.\n"
            "{}\n"
            "Write a professional, clear, one-sentence bullet summarizing the covenant.\n"
            "Bullet:"
        ),

        "if_false_summary": "No ordinary course covenant is included in the agreement.",
        "if_missing_summary": "Ordinary course covenant summary is missing.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "bullet",
        "max_words": 60,
        "summary_type": "concise",
    },

    "Ordinary Course - Fulsome": {
        "section": "ordinary_course",                         
        "subsection": "ordinary_course_covenant",             

        "summary_field": "concise_standard_summary",
        "reference_field": "reference_section",
        "output_source_field": "clause_text",

        "prompt_template": (
            "You are a legal analyst reviewing a merger agreement covenant. Below is the full text of a clause:\n\n"
            "{}\n\n"
            "Your task is to write a precise, paraphrased 1-paragraph summary of the covenant that:\n"
            "- States the main obligation (e.g., use of commercially reasonable efforts to operate in ordinary course)\n"
            "- Clearly lists the exceptions or carve-outs (e.g., required by law, agreed in writing, etc.)\n"
            "- Includes any specific operational duties or examples mentioned (e.g., preserving relationships, asset protection)\n"
            "- Uses some key quoted phrases from the clause if helpful, but avoid copying large sections\n"
            "- Avoids generic phrasing — aim to capture the nuance of the original clause.\n\n"
            "Write only one paragraph."
        ),

        "if_false_summary": "No detailed clause for ordinary course covenant found.",
        "if_missing_summary": "Detailed clause text for ordinary course is missing.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "paragraph",
        "max_words": 150,
        "summary_type": "fulsome",
    },
}
