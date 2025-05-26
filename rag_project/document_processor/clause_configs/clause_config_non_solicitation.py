NON_SOLICITATION_CLAUSE_CONFIG = {
    # =======================
    # 🔹 Match Right
    # =======================
    "Match Right - Concise": {
        "section": "non_solicitation",
        "subsection": "match_right",
        "included_field": "match_right_initial_included",
        "summary_field": "match_right_initial_explanation",
        "reference_field": "reference_section",
        "output_source_field": "answer",

        "prompt_template": (
            "If the text below describes a right-to-match period in a merger agreement, summarize it with the number "
            "of days (in digits) and who holds the right. Format like:\n"
            "\"Parent has a 4 Business Day right-to-match.\"\n\nText:\n{}"
        ),

        "if_false_summary": "A right-to-match was not included in the merger contract.",
        "if_missing_summary": "Right-to-match information is not available.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "bullet",
        "max_words": 50,
        "summary_type": "concise",
    },

    # =======================
    # 🔹 Go Shop Clause
    # =======================
    "Go Shop - Concise": {
        "section": "non_solicitation",
        "subsection": "go_shop",
        "included_field": "go_shop_included",
        "summary_field": "go_shop_explanation",
        "reference_field": "reference_section",
        "output_source_field": "answer",

        "prompt_template": (
            "Summarize this go-shop clause in 1–2 sentences. Focus on whether the Company may solicit "
            "competing proposals after signing and any restrictions.\n\nText:\n{}"
        ),

        "if_false_summary": "There is no go-shop or window-shop provision.",
        "if_missing_summary": "Go-shop information is not available.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "bullet",
        "max_words": 60,
        "summary_type": "concise",
    },

    # =======================
    # 🔹 Notice of Competing Offer
    # =======================
    "Notice of Competing Offer - Concise": {
        "section": "non_solicitation",
        "subsection": "notice_of_competing_offer",
        "included_field": "notice_of_competing_offer_included",
        "summary_field": "notice_of_competing_offer_explanation",
        "reference_field": "reference_section",
        "output_source_field": "answer",

        "prompt_template": (
            "Summarize the notice requirements below. Focus on what the Company must notify Parent about, "
            "the timing (e.g., 24 hours), and the types of proposals or discussions covered.\n\nText:\n{}"
        ),

        "if_false_summary": "The merger contract did not specify any details related to a Notice of Competing Offer.",
        "if_missing_summary": "Notice of competing offer information is not available.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "bullet",
        "max_words": 80,
        "summary_type": "concise",
    },

    # =======================
    # 🔹 Ongoing Update
    # =======================
    "Ongoing Update - Concise": {
        "section": "non_solicitation",
        "subsection": "ongoing_update",
        "included_field": "ongoing_update_included",
        "summary_field": "ongoing_update_explanation",
        "reference_field": "reference_section",
        "output_source_field": "answer",

        "prompt_template": (
            "Summarize the ongoing update requirement. Focus on how frequently the Company must update Parent "
            "(e.g., within 24 hours) and about what kinds of developments.\n\nText:\n{}"
        ),

        "if_false_summary": "Ongoing update requirement information is not available.",
        "if_missing_summary": "Ongoing update clause was referenced, but no explanation was provided.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "bullet",
        "max_words": 80,
        "summary_type": "concise",
    },

    # =======================
    # 🔹 Superior Proposal Engagement
    # =======================
    "Superior Proposal Engagement - Concise": {
        "section": "non_solicitation",
        "subsection": "superior_proposal_engagement",
        "included_field": "superior_proposal_engagement_included",
        "summary_field": "superior_proposal_engagement_explanation",
        "reference_field": "reference_section",
        "output_source_field": "answer",

        "prompt_template": (
            "Summarize the engagement procedure when a Superior Proposal is received. "
            "Describe notice obligations to Parent, initial match rights (e.g., 4 Business Days), "
            "and any follow-on rights if the Superior Proposal changes.\n\nText:\n{}"
        ),

        "if_false_summary": "No details available related to Superior Offers.",
        "if_missing_summary": "Superior proposal engagement clause was referenced, but no explanation was provided.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "paragraph",
        "max_words": 150,
        "summary_type": "fulsome",
    },

    # =======================
    # 🔹 Adverse Recommendation - Superior Proposal
    # =======================
    "Adverse Recommendation (Superior Proposal) - Concise": {
        "section": "non_solicitation",
        "subsection": "adverse_recommendation_change_superior_proposal",
        "included_field": "adverse_recommendation_change_superior_proposal_included",
        "summary_field": "adverse_recommendation_change_superior_proposal_explanation",
        "reference_field": "reference_section",
        "output_source_field": "answer",

        "prompt_template": (
            "Summarize the Company Board's ability to change its recommendation due to a Superior Proposal. "
            "Explain the fiduciary duty standard, good faith requirements, and any timing or notice obligations.\n\nText:\n{}"
        ),

        "if_false_summary": "No adverse recommendation change provision related to a Superior Proposal was included in the merger contract.",
        "if_missing_summary": "Adverse recommendation change provision was referenced, but no explanation was provided.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "paragraph",
        "max_words": 150,
        "summary_type": "fulsome",
    },

    # =======================
    # 🔹 Adverse Recommendation - Intervening Event
    # =======================
    "Adverse Recommendation (Intervening Event) - Concise": {
        "section": "non_solicitation",
        "subsection": "adverse_recommendation_change_intervening_event",
        "included_field": "adverse_recommendation_change_intervening_event_included",
        "summary_field": "adverse_recommendation_change_intervening_event_explanation",
        "reference_field": "reference_section",
        "output_source_field": "answer",

        "prompt_template": (
            "Summarize the Company Board's right to change its recommendation due to an Intervening Event. "
            "Include fiduciary duty language, good faith standards, and notice obligations.\n\nText:\n{}"
        ),

        "if_false_summary": "No adverse recommendation change provision related to an Intervening Event was included in the merger contract.",
        "if_missing_summary": "Intervening event recommendation change clause was referenced, but no explanation was provided.",

        "use_llm": True,
        "use_short_reference": True,
        "format_style": "paragraph",
        "max_words": 150,
        "summary_type": "fulsome",
    },
}
