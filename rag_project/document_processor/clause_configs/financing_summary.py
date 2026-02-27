FINANCING_SUMMARY_CLAUSE_CONFIG = {
    # =======================
    # 🔹 Committed Financing
    # =======================
    "Committed Financing - Concise": {
        "section": "financing",
        "subsection": "committed_financing",
        "included_field": "committed_financing_present",
        "summary_field": "committed_financing_summary",
        "reference_field": "reference_section",
        "output_source_field": "answer",

        "prompt_template": (
            "Summarize the Committed Financing below. Focus on Parent received committed financing from Bank.\n\nText:\n{}"
        ),

        "if_false_summary": "Committed financing was not included in the merger contract.",
        "if_missing_summary": "Committed financing information is not available.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "bullet",
        "max_words": 50,
        "summary_type": "concise",
    },
    "Committed Financing - Fulsome": {
        "section": "financing",
        "subsection": "committed_financing",
        "included_field": "committed_financing_present",
        "summary_field": "committed_financing_explanation",
        "reference_field": "reference_section",
        "output_source_field": "clause_text",

        "prompt_template": (
            "You are a legal analyst reviewing a merger agreement Committed Financing. Below is the full text of a clause:\n\n"
            "{}\n\n"
            "Your task is to write a precise, paraphrased 1-paragraph summary of the Committed Financing that:\n"
            "- Focus on Parent received committed financing from Bank\n"
            "- Uses some key quoted phrases from the clause if helpful, but avoid copying large sections\n"
            "- Avoids generic phrasing — aim to capture the nuance of the original clause.\n\n"
            "Write only one paragraph."
        ),

        "if_false_summary": "No detailed clause for committed financing found.",
        "if_missing_summary": "Detailed clause text for committed financing is missing.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "paragraph",
        "max_words": 150,
        "summary_type": "fulsome",
    },
    "Cash On Hand Financing - Fulsome": {
        "section": "financing",
        "subsection": "cash_on_hand",
        "included_field": "cash_on_hand_specified_present",
        "summary_field": "cash_on_hand_specified",
        "reference_field": "reference_section",
        "output_source_field": "clause_text",

        "prompt_template": (
            "You are a legal analyst reviewing a merger agreement Cash On Hand Financing. Below is the full text of a clause:\n\n"
            "{}\n\n"
            "Your task is to write a precise, paraphrased 1-paragraph summary of the Cash On Hand Financing that:\n"
            "- Uses some key quoted phrases from the clause if helpful, but avoid copying large sections\n"
            "- Avoids generic phrasing — aim to capture the nuance of the original clause.\n\n"
            "Write only one paragraph."
        ),

        "if_false_summary": "No detailed clause for committed financing found.",
        "if_missing_summary": "Detailed clause text for committed financing is missing.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "paragraph",
        "max_words": 150,
        "summary_type": "fulsome",
    },
    "Financing Efforts Summary - Fulsome": {
        "section": "financing",
        "subsection": "financing_efforts_summary",
        "included_field": "",
        "summary_field": "financing_efforts_summary",
        "reference_field": "reference_section",
        "output_source_field": "clause_text",

        "prompt_template": (
            "You are a legal analyst reviewing a merger agreement Financing Efforts. Below is the full text of a clause:\n\n"
            "{}\n\n"
            "Your task is to write a precise, paraphrased 1-paragraph summary of the Financing Efforts that:\n"
            "- Uses some key quoted phrases from the clause if helpful, but avoid copying large sections\n"
            "- Avoids generic phrasing — aim to capture the nuance of the original clause.\n\n"
            "Write only one paragraph."
        ),

        "if_false_summary": "No detailed clause for committed financing found.",
        "if_missing_summary": "Detailed clause text for committed financing is missing.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "paragraph",
        "max_words": 150,
        "summary_type": "fulsome",
    },
    "Substitute Financing Notice And Efforts - Fulsome": {
        "section": "financing",
        "subsection": "substitute_financing_notice_and_efforts",
        "included_field": "",
        "summary_field": "substitute_financing_notice_and_efforts",
        "reference_field": "reference_section",
        "output_source_field": "clause_text",

        "prompt_template": (
            "You are a legal analyst reviewing a merger agreement Substitute Financing Notice And Efforts. Below is the full text of a clause:\n\n"
            "{}\n\n"
            "Your task is to write a precise, paraphrased 1-paragraph summary of the Substitute Financing Notice And Efforts that:\n"
            "- Uses some key quoted phrases from the clause if helpful, but avoid copying large sections\n"
            "- Avoids generic phrasing — aim to capture the nuance of the original clause.\n\n"
            "Write only one paragraph."
        ),

        "if_false_summary": "No detailed clause for committed financing found.",
        "if_missing_summary": "Detailed clause text for committed financing is missing.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "paragraph",
        "max_words": 150,
        "summary_type": "fulsome",
    },

}
