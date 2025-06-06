import openai
import concurrent.futures
import os
import logging

logger = logging.getLogger(__name__)


class SummaryEngineUtil:

    def __init__(self):
        # Set up executor for background tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

        # Set up openAi cleint
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"))

    # =========================
    # LLM FUNCTION FOR GENERATING BULLET POINTS
    # =========================
    def generate_bullet(self, prompt, temperature=0.3, max_tokens=300):
        try:

            self.openai_client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"))
            # self.MAX_METADATA_SIZE = 40960

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal analyst summarizing merger agreement clauses.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return "" + response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"⚠️ LLM error: {e}")
            return "Error generating summary."

    # =========================
    # CLAUSE LOOKUP UTIL
    # =========================
    def get_items(self, schema, section, subsection):
        section_data = schema.get(section, {})
        if subsection:
            section_data = section_data.get(subsection, {})

        if isinstance(section_data, list):
            return section_data  # ✅ Flat list structure
        elif isinstance(section_data, dict):
            # ✅ Nested dict with a list inside (grab the first list)
            possible_lists = [
                v for v in section_data.values() if isinstance(v, list)]
            if possible_lists:
                return possible_lists[0]
            else:
                return []
        else:
            return []

    # =========================
    # CLAUSE SUMMARIZER ENGINE
    # =========================
    def summarize_from_config(self, schema, config):
        section = config.get("section")
        subsection = config.get("subsection")

        # ----- Pull data -----
        items = self.get_items(schema, section, subsection)
        print(
            f"DEBUG: Found items → {[item.get('field_name') for item in items]}")

        fields = {
            item.get("field_name"): item for item in items if "field_name" in item
        }

        summary_field = config["summary_field"]
        source_item = fields.get(summary_field, {})

        if not source_item:
            return f" {config['if_missing_summary']}"

        # ----- Clause reference -----
        reference = source_item.get(config.get(
            "reference_field", "reference_section"))

        # ----- Clause text -----
        output_source_field = config.get("output_source_field", None)

        if output_source_field:
            clause_text = source_item.get(output_source_field)
        else:
            clause_text = source_item.get(
                "clause_text") or source_item.get("answer")

        logger.info(f"DEBUG: clause_text → {clause_text}")
        logger.info(f"DEBUG: reference → {reference}")

        # ----- Handle inclusion check -----
        included_field = config.get("included_field")
        included = True  # Default to True if no inclusion field

        if included_field:
            included_value_data = fields.get(included_field, "")
            included_value = included_value_data.get("answer")
            if isinstance(included_value, bool):
                included = included_value
            elif isinstance(included_value, str):
                included = included_value.strip().lower() in ("true", "yes")
            else:
                included = False

        logger.info(f"DEBUG: included → {included_field}")
        logger.info(f"DEBUG: Soruce Item → {source_item}")
        logger.info(
            f"DEBUG: included_value → {source_item.get(included_field, '')}")

        # ----- Handle exclusion -----
        if not included:
            return f"{config['if_false_summary']}"

        if not clause_text:
            return f"{config['if_missing_summary']}"

        # ----- Skip LLM -----
        if not config.get("use_llm", True):
            bullet = f" {clause_text}"
            return bullet + (f"\n   {reference}" if reference else "")

        # ----- Format reference -----
        short_ref = (
            self.shorten_reference(reference)
            if (reference and config.get("use_short_reference"))
            else reference
        )

        # ----- LLM Prompt -----
        prompt = config["prompt_template"].format(clause_text)

        max_words = config.get("max_words")
        max_tokens = (
            300 if not max_words else int(max_words * 1.5)
        )  # ~1.5 tokens per word

        bullet = self.generate_bullet(prompt, max_tokens=max_tokens)

        # ----- Apply format -----
        if config.get("format_style", "bullet") == "paragraph":
            bullet = bullet.lstrip("").strip()

        if short_ref:
            bullet += f"\n  • {short_ref}"

        return bullet

    # =========================
    # REFERENCE SHORTENER
    # =========================
    def shorten_reference(self, ref):
        if not ref:
            return None
        parts = ref.split(">")
        for part in reversed(parts):
            candidate = part.strip()
            if any(x in candidate for x in ["Section", "§", "5.", "4.", "6.", "7."]):
                return candidate
        return parts[-1].strip() if parts else ref
