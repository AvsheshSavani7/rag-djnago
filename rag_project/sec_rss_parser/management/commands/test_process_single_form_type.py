from django.core.management.base import BaseCommand
from sec_rss_parser.services import SECFeedProcessor, FORM_TYPES
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Test _process_single_form_type by passing a form_type (e.g. 8-k, DEFM14A, 10-K).'

    def add_arguments(self, parser):
        parser.add_argument(
            'form_type',
            type=str,
            help=f'SEC form type to process. Examples: {", ".join(FORM_TYPES[:5])}...',
        )

    def handle(self, *args, **options):
        form_type = options['form_type'].strip()

        self.stdout.write(f'Testing _process_single_form_type for form_type={form_type!r}...')

        try:
            processor = SECFeedProcessor(form_type=form_type)
            result = processor._process_single_form_type(form_type)

            if result['success']:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Success: form_type={result['form_type']}, "
                        f"processed={len(result['processed_items'])}, "
                        f"new_saved={result['new_items_count']}"
                    )
                )
                if result['processed_items']:
                    for i, item in enumerate(result['processed_items'][:5], 1):
                        acc = item.get('accession_number', 'N/A')
                        company = item.get('company_name', 'N/A')
                        self.stdout.write(f"  {i}. {company} ({acc})")
                    if len(result['processed_items']) > 5:
                        self.stdout.write(f"  ... and {len(result['processed_items']) - 5} more")
            else:
                self.stdout.write(
                    self.style.ERROR(
                        f"Failed: {result.get('error', 'Unknown error')}"
                    )
                )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Exception: {e}")
            )
            logger.exception("Error in test_process_single_form_type")
