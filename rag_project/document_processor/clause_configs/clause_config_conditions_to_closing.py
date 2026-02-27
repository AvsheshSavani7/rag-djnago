CLOSING_CONDITIONS_CLAUSE_CONFIG = {
    "Mutual Conditions - Concise": {
        "section": "conditions_to_closing",
        "subsection": "",
        "included_field": "",
        "summary_field": "mutual_conditions_list",
        "reference_field": "reference_section",
        "output_source_field": "answer",
        "prompt_template": (
            "Below are mutual closing conditions from a merger agreement:\n\n"
            "{}\n\n"
            "Summarize as a single sentence: 'Mutual conditions include: ...' Use M&A terms (e.g., Stockholder Approval, HSR Clearance, Form F-4)."
        ),

        "if_false_summary": "No mutual conditions are included in the agreement.",
        "if_missing_summary": "No mutual conditions found.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "bullet",
        "max_words": 50,
        "summary_type": "concise",
    },

    "Parent Conditions - Concise": {
        "section": "conditions_to_closing",
        "subsection": "",
        "included_field": "",
        "summary_field": "parent_closing_conditions",
        "reference_field": "reference_section",
        "output_source_field": "answer",
        "prompt_template": (
            "Conditions to Parent's obligation to close:\n\n"
            "{}\n\n"
            "Summarize in one sentence using M&A terms like 'Company reps and warranties true and correct', 'no Company MAE', 'compliance with obligations', 'officer's certificate'."
        ),

        "if_false_summary": "No Parent closing conditions are included in the agreement.",
        "if_missing_summary": "No Parent closing conditions found.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "bullet",
        "max_words": 50,
        "summary_type": "concise",
    },

    "Company Conditions - Concise": {
        "section": "conditions_to_closing",
        "subsection": "",
        "included_field": "",
        "summary_field": "company_closing_conditions",
        "reference_field": "reference_section",
        "output_source_field": "answer",
        "prompt_template": (
            "Conditions to Company's obligation to close:\n\n"
            "{}\n\n"
            "Summarize in one sentence using M&A terms like 'Parent reps and warranties true and correct', 'no Parent MAE', 'compliance with obligations', 'officer's certificate'."
        ),

        "if_false_summary": "No Company closing conditions are included in the agreement.",
        "if_missing_summary": "No Company closing conditions found.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "bullet",
        "max_words": 50,
        "summary_type": "concise",
    },

    "Shareholder Approval - Concise": {
        "section": "conditions_to_closing",
        "subsection": "",
        "included_field": "target_shareholder_approval_required",
        "summary_field": "target_shareholder_approval_threshold",
        "reference_field": "reference_section",
        "output_source_field": "answer",

        "prompt_template": (
            "Conditions to Company's obligation to close:\n\n"
            "{}\n\n"
            "Summarize in one sentence using M&A terms like 'Parent reps and warranties true and correct', 'no Parent MAE', 'compliance with obligations', 'officer's certificate'."
        ),

        "if_false_summary": "No Company closing conditions are included in the agreement.",
        "if_missing_summary": "No Company closing conditions found.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "bullet",
        "max_words": 50,
        "summary_type": "concise",
    },
    "HSR Clearance - Concise": {
        "section": "conditions_to_closing",
        "subsection": "",
        "included_field": "",
        "summary_field": "hsr_clearance_required",
        "reference_field": "reference_section",
        "output_source_field": "answer",

        "prompt_template": (
            "Conditions to Company's obligation to close:\n\n"
            "{}\n\n"
            "Summarize in one sentence using M&A terms like 'Parent reps and warranties true and correct', 'no Parent MAE', 'compliance with obligations', 'officer's certificate'."
        ),

        "if_false_summary": "No HSR Clearance is required.",
        "if_missing_summary": "No HSR Clearance is required.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "bullet",
        "max_words": 50,
        "summary_type": "concise",
    },
    "Other Customary Conditions - Concise": {
        "section": "conditions_to_closing",
        "subsection": "",
        "included_field": "",
        "summary_field": "other_customary_conditions",
        "reference_field": "reference_section",
        "output_source_field": "answer",

        "prompt_template": (
            "Conditions to Company's obligation to close:\n\n"
            "{}\n\n"
            "Summarize in one sentence using M&A terms like 'Parent reps and warranties true and correct', 'no Parent MAE', 'compliance with obligations', 'officer's certificate'."
        ),

        "if_false_summary": "No other customary conditions are included in the agreement.",
        "if_missing_summary": "No other customary conditions found.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "bullet",
        "max_words": 50,
        "summary_type": "concise",
    },

    "Additional Customary Conditions - Concise": {

        "is_grouping_of_fields": True,
        "grouping_of_fields": [

            {
                "included_field": "",
                "summary_field": "other_customary_conditions",
                "reference_field": "reference_section",
                "output_source_field": "answer"
            }

        ],



    }
}
