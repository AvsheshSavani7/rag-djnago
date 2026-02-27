"""
One-off management command to list or remove duplicate SEC filings by accession_number.

By default only LISTS duplicates (no deletion). Use --delete to actually remove.
Run before deploying unique index on SECFiling.accession_number.

Uses raw PyMongo collection (no SECFiling._get_collection()) so index creation
is not triggered, avoiding conflict with existing non-unique accession_number index.
"""
from datetime import datetime as dt
import json
from django.core.management.base import BaseCommand
from sec_rss_parser.models import SECFiling


# Collection name for sec_filings (must match SECFiling.meta)
SEC_FILINGS_COLLECTION = 'sec_filings'


def get_sec_filings_collection():
    """Get raw PyMongo collection without triggering MongoEngine ensure_indexes()."""
    db = SECFiling._get_db()
    return db[SEC_FILINGS_COLLECTION]


class Command(BaseCommand):
    help = (
        'List duplicate SEC filings by accession_number (default: list only, no delete). '
        'Use --delete to remove duplicates (keeps earliest created_at per accession). '
        'Use --output FILE to save the list to a file.'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--delete',
            action='store_true',
            help='Actually delete duplicate documents (default is list only).',
        )
        parser.add_argument(
            '--output',
            type=str,
            metavar='FILE',
            help='Write duplicate list to this file (JSON).',
        )

    def handle(self, *args, **options):
        do_delete = options['delete']
        output_file = options.get('output')

        collection = get_sec_filings_collection()

        # Find accession_numbers that appear more than once
        pipeline = [
            {'$group': {
                '_id': '$accession_number',
                'count': {'$sum': 1},
                'docs': {'$push': {'id': '$_id', 'created_at': '$created_at'}},
            }},
            {'$match': {'count': {'$gt': 1}}},
        ]
        cursor = collection.aggregate(pipeline)

        duplicates = list(cursor)
        if not duplicates:
            self.stdout.write(self.style.SUCCESS(
                'No duplicate accession_numbers found.'))
            return

        total_duplicate_groups = len(duplicates)
        total_to_delete = sum(d['count'] - 1 for d in duplicates)
        self.stdout.write(
            f'Found {total_duplicate_groups} accession_numbers with duplicates '
            f'({total_to_delete} extra documents to remove).'
        )
        self.stdout.write('')

        ids_to_delete = []
        list_data = []

        for dup in duplicates:
            accession = dup['_id']
            docs = dup['docs']
            # Sort by created_at ascending; keep first, delete rest (None -> last)

            def _sort_key(x):
                c = x.get('created_at')
                return c if c is not None else dt.datetime.max
            docs_sorted = sorted(docs, key=_sort_key)
            keep_id = docs_sorted[0]['id']
            delete_ids = [str(d['id']) for d in docs_sorted[1:]]
            ids_to_delete.extend(delete_ids)
            # For JSON, use string ids
            list_data.append({
                'accession_number': accession,
                'count': dup['count'],
                'keep_id': str(keep_id),
                'delete_ids': delete_ids,
            })
            self.stdout.write(
                f'  {accession}: keep 1 (id={keep_id}), delete {len(delete_ids)}: {delete_ids}'
            )

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(list_data, f, indent=2)
            self.stdout.write(self.style.SUCCESS(
                f'\nList written to: {output_file}'))

        if not do_delete:
            self.stdout.write(
                self.style.WARNING(
                    f'\nList only (no documents deleted). To delete these {len(ids_to_delete)} document(s) run with --delete.'
                )
            )
            return

        deleted = 0
        for doc_id in ids_to_delete:
            try:
                # Use raw collection delete to avoid triggering ensure_indexes()
                result = collection.delete_one({'_id': doc_id})
                if result.deleted_count:
                    deleted += 1
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error deleting id={doc_id}: {e}')
                )

        self.stdout.write(
            self.style.SUCCESS(f'Deleted {deleted} duplicate document(s).')
        )
