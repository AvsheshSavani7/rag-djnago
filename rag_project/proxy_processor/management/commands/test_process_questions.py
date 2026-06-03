"""
Management command to test _process_questions from ProxySummaryServiceV2.

Usage:
  python manage.py test_process_questions --sec_filing_summary_id=<your_id>
  python manage.py test_process_questions --sec_filing_summary_id=2abb4c92-037e-4a58-a016-59122d401533

Outputs:
  - combined_results.txt  (all combined chunks passed to Claude for each question)
  - all_summaries.txt     (final summaries output)
"""
import os
import json
import logging

from django.core.management.base import BaseCommand
from sec_rss_parser.proxy_summary_service_v2 import ProxySummaryServiceV2
from proxy_processor.arb_summary_doc_new_02_Dec_25 import QueryProcessor

logger = logging.getLogger(__name__)


def _resolve_questions_file():
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "quetions.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "..", "proxy_processor", "quetions.json"),
    ]
    for path in possible_paths:
        normalized = os.path.normpath(path)
        if os.path.exists(normalized):
            return normalized
    return None


class Command(BaseCommand):
    help = "Test _process_questions: fetches chunks, filters by title, combines with vector search, calls Claude."

    def add_arguments(self, parser):
        parser.add_argument(
            "--sec_filing_summary_id",
            type=str,
            required=True,
            help="The sec_filing_summary_id to fetch chunks for.",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default=".",
            help="Directory to write output files (default: current directory).",
        )

    def handle(self, *args, **options):
        sec_filing_summary_id = options["sec_filing_summary_id"]
        output_dir = options["output_dir"]

        self.stdout.write(
            f"Testing _process_questions for sec_filing_summary_id: {sec_filing_summary_id}")

        questions_file = _resolve_questions_file()
        if not questions_file:
            self.stderr.write(self.style.ERROR("Could not find quetions.json"))
            return

        self.stdout.write(f"Using questions file: {questions_file}")

        with open(questions_file, "r", encoding="utf-8") as f:
            questions_data = json.load(f)

        self.stdout.write(f"Loaded {len(questions_data)} questions")

        service = ProxySummaryServiceV2()
        processor = service._create_query_processor()

        if processor is None:
            self.stderr.write(self.style.ERROR(
                "Failed to initialize QueryProcessor"))
            return

        # Fetch base context chunks filtered by title
        title_filters = [
            "QUESTIONS AND ANSWERS",
            "QUESTIONS AND ANSWERS ABOUT THE SPECIAL MEETING AND THE MERGER",
            "QUESTIONS AND ANSWERS ABOUT THE SPECIAL MEETING",
            "QUESTIONS AND ANSWERS ABOUT THE PROPOSALS AND THE SPECIAL MEETING",
            "QUESTIONS AND ANSWERS ABOUT THE MERGER",
            "QUESTIONS AND ANSWERS ABOUT THE MEETINGS",
        ]

        self.stdout.write("Fetching all chunks for sec_filing_summary_id...")
        base_context_chunks = service._fetch_chunks_by_title_filter(
            processor, sec_filing_summary_id, title_filters
        )
        self.stdout.write(self.style.SUCCESS(
            f"Fetched {len(base_context_chunks)} base context chunks by title filter"
        ))

        # Print matched titles
        matched_titles = set(chunk.get('title', '')
                             for chunk in base_context_chunks)
        self.stdout.write(f"Matched titles: {matched_titles}")

        all_combined_results_output = []
        all_summaries = []

        for question_key, question_data in questions_data.items():
            display_question, prompt_question = service._resolve_question_entry(
                question_data
            )
            if not prompt_question:
                self.stdout.write(self.style.WARNING(
                    f"Skipping {question_key}: empty prompt/display"))
                continue

            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"Processing: {question_key}")
            self.stdout.write(f"Display: {display_question}")
            prompt_preview = (
                prompt_question[:120] + "..."
                if len(prompt_question) > 120
                else prompt_question
            )
            self.stdout.write(f"Prompt: {prompt_preview}")

            try:
                results = service._search_chunks_by_filing_id(
                    processor, prompt_question, sec_filing_summary_id)
                self.stdout.write(
                    f"  Vector search returned {len(results)} chunks")

                combined_results = service._deduplicate_chunks(
                    base_context_chunks + results
                )
                self.stdout.write(
                    f"  Combined (deduplicated): {len(combined_results)} chunks")

                # Log combined results for this question
                all_combined_results_output.append(f"{'='*60}")
                all_combined_results_output.append(f"QUESTION: {question_key}")
                all_combined_results_output.append(
                    f"DISPLAY: {display_question}")
                all_combined_results_output.append(
                    f"PROMPT: {prompt_question}")
                all_combined_results_output.append(
                    f"TOTAL CHUNKS: {len(combined_results)}")
                all_combined_results_output.append(f"{'='*60}")
                for i, chunk in enumerate(combined_results):
                    all_combined_results_output.append(
                        f"\n--- Chunk {i+1} (id: {chunk['id']}, title: {chunk.get('title', 'N/A')}) ---")
                    all_combined_results_output.append(
                        chunk.get('text', ''))
                all_combined_results_output.append("\n\n")

                if not combined_results:
                    self.stdout.write(self.style.WARNING(
                        f"  No results for {question_key}, skipping"))
                    continue

                # Get Claude response
                self.stdout.write("  Calling Claude...")
                claude_answer = processor.get_claude_response(
                    prompt_question, combined_results)
                self.stdout.write(
                    f"  Claude response: {len(claude_answer)} chars")

                # Generate arb summary
                arb_summary = processor.generate_arb_summary(
                    claude_answer, display_question)
                all_summaries.append(arb_summary)
                self.stdout.write(self.style.SUCCESS(
                    f"  Summary generated: {len(arb_summary)} chars"))

            except Exception as e:
                self.stderr.write(self.style.ERROR(
                    f"  Error processing {question_key}: {str(e)}"))
                continue

        # Write combined_results to file
        combined_file = os.path.join(output_dir, "combined_results.txt")
        with open(combined_file, "w", encoding="utf-8") as f:
            f.write("\n".join(all_combined_results_output))
        self.stdout.write(self.style.SUCCESS(
            f"\nWrote combined results to: {combined_file}"))

        # Write all_summaries to file
        summaries_file = os.path.join(output_dir, "all_summaries.txt")
        with open(summaries_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_summaries))
        self.stdout.write(self.style.SUCCESS(
            f"Wrote all summaries to: {summaries_file}"))

        self.stdout.write(self.style.SUCCESS(
            f"\nDone! Processed {len(all_summaries)} questions successfully."))
