from django.core.management.base import BaseCommand
from sec_rss_parser.services import SECFeedProcessor
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Process SEC RSS feed and store new filings'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Run without saving to database',
        )

    def handle(self, *args, **options):
        self.stdout.write('Starting SEC RSS feed processing...')

        try:
            processor = SECFeedProcessor()
            result = processor.process_feed()

            if result['success']:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Success: {result['message']}"
                    )
                )
                self.stdout.write(f"Total items: {result['total_items']}")
                self.stdout.write(f"New items: {result['new_items']}")
            else:
                self.stdout.write(
                    self.style.ERROR(
                        f"Error: {result['error']}"
                    )
                )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Exception occurred: {e}")
            )
            logger.error(f"Error in process_sec_feed command: {e}")
