"""
Management command to backfill deal_id on existing EightKSummary and Ex99_1Summary records.

For each record, looks up a deal by CIK (target then acquirer) with status Open/Unknown.
If found, sets deal_id; if not found, leaves deal_id null.

Usage:
  python manage.py backfill_8k_99_1_deal_id
  python manage.py backfill_8k_99_1_deal_id --dry-run
  python manage.py backfill_8k_99_1_deal_id --collection 8k
  python manage.py backfill_8k_99_1_deal_id --collection 99_1
"""
from django.core.management.base import BaseCommand
from document_processor.models import ProcessingJob

from sec_rss_parser.models import EightKSummary, Ex99_1Summary
from sec_rss_parser.services import DEAL_STATUS_OPEN_OR_UNKNOWN, normalize_cik


def find_deal_id_for_cik(cik_number):
    """Return deal id (str) if CIK matches a deal's target or acquirer, else None."""
    if not cik_number:
        return None
    cik_normalized = normalize_cik(cik_number)
    try:
        deal = ProcessingJob.objects(
            cik=cik_normalized,
            deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
        ).first()
        if deal:
            return str(deal.id)
        deal = ProcessingJob.objects(
            acquirer_cik=cik_normalized,
            deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
        ).first()
        if deal:
            return str(deal.id)
    except Exception:
        pass
    return None


class Command(BaseCommand):
    help = (
        'Backfill deal_id on all existing 8k_summary and 99_1_summary records. '
        'Uses CIK to match target or acquirer; leaves deal_id null if no match.'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Only report what would be updated; do not save.',
        )
        parser.add_argument(
            '--collection',
            type=str,
            choices=['8k', '99_1', 'all'],
            default='all',
            help='Which collection(s) to process: 8k, 99_1, or all (default).',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        collection = options['collection']

        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN — no changes will be saved.'))

        grand_total_updated = 0

        # 8-K summaries
        if collection in ('8k', 'all'):
            total_updated = total_null = total_unchanged = 0
            self.stdout.write('Processing 8k_summary (EightKSummary)...')
            for doc in EightKSummary.objects.all():
                deal_id = find_deal_id_for_cik(doc.cik_number)
                if deal_id is not None:
                    if doc.deal_id != deal_id:
                        if not dry_run:
                            doc.deal_id = deal_id
                            doc.save()
                        total_updated += 1
                    else:
                        total_unchanged += 1
                else:
                    total_null += 1
            self.stdout.write(
                self.style.SUCCESS(
                    f'  8k_summary: updated={total_updated}, left_null={total_null}, unchanged={total_unchanged}'
                )
            )
            grand_total_updated += total_updated

        # EX-99.1 summaries
        if collection in ('99_1', 'all'):
            total_updated = total_null = total_unchanged = 0
            self.stdout.write('Processing 99_1_summary (Ex99_1Summary)...')
            for doc in Ex99_1Summary.objects.all():
                deal_id = find_deal_id_for_cik(doc.cik_number)
                if deal_id is not None:
                    if doc.deal_id != deal_id:
                        if not dry_run:
                            doc.deal_id = deal_id
                            doc.save()
                        total_updated += 1
                    else:
                        total_unchanged += 1
                else:
                    total_null += 1
            self.stdout.write(
                self.style.SUCCESS(
                    f'  99_1_summary: updated={total_updated}, left_null={total_null}, unchanged={total_unchanged}'
                )
            )
            grand_total_updated += total_updated

        self.stdout.write(
            self.style.SUCCESS(
                f'Done. Total records updated: {grand_total_updated}' + (' (dry-run)' if dry_run else '')
            )
        )
