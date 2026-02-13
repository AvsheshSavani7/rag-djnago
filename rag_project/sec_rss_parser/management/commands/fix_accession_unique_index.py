"""
One-off management command to fix IndexKeySpecsConflict on accession_number.

MongoDB has an existing non-unique index "accession_number_1", while MongoEngine
tries to create a unique index with the same name. This command drops the old
index so the app can create the correct unique index on next run.

Run once on the server, then restart the feed processor.

  python manage.py fix_accession_unique_index

Optional: --recreate will call ensure_indexes() after dropping so the unique
index is created immediately (without waiting for next save).

If you have duplicate accession_numbers in sec_filings, run
  python manage.py dedupe_sec_filings --delete
before creating the unique index, or index creation will fail.
"""
import logging
from django.core.management.base import BaseCommand

from sec_rss_parser.models import SECFiling

logger = logging.getLogger(__name__)

# Collections that have accession_number with unique=True in models
COLLECTIONS = [
    ('sec_filings', 'SECFiling')
]
INDEX_NAME = 'accession_number_1'


class Command(BaseCommand):
    help = (
        'Drop existing non-unique index accession_number_1 so MongoEngine can '
        'create the unique index. Use --recreate to build the unique index immediately.'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--recreate',
            action='store_true',
            help='After dropping, call ensure_indexes() so the unique index is created now.',
        )

    def handle(self, *args, **options):
        do_recreate = options['recreate']
        db = SECFiling._get_db()

        for coll_name, doc_label in COLLECTIONS:
            coll = db[coll_name]
            try:
                indexes = list(coll.list_indexes())
            except Exception as e:
                self.stdout.write(self.style.WARNING(
                    f'Could not list indexes for {coll_name}: {e}'
                ))
                continue

            for idx in indexes:
                if idx.get('name') == INDEX_NAME:
                    is_unique = idx.get('unique', False)
                    if is_unique:
                        self.stdout.write(
                            f'{doc_label} ({coll_name}): index {INDEX_NAME} is already unique, nothing to do.'
                        )
                    else:
                        try:
                            coll.drop_index(INDEX_NAME)
                            self.stdout.write(self.style.SUCCESS(
                                f'{doc_label} ({coll_name}): dropped non-unique index {INDEX_NAME}.'
                            ))
                        except Exception as e:
                            self.stdout.write(self.style.ERROR(
                                f'{doc_label} ({coll_name}): failed to drop index: {e}'
                            ))
                    break
            else:
                self.stdout.write(
                    f'{doc_label} ({coll_name}): no index named {INDEX_NAME} found.'
                )

        if do_recreate:
            self.stdout.write('Recreating indexes via MongoEngine...')
            try:
                from sec_rss_parser.models import AccessionLookedUp
                SECFiling.ensure_indexes()
                AccessionLookedUp.ensure_indexes()
                self.stdout.write(self.style.SUCCESS(
                    'ensure_indexes() completed.'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(
                    f'ensure_indexes failed: {e}'))
