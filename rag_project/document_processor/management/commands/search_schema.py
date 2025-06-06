from django.core.management.base import BaseCommand
from document_processor.services import SchemaCategorySearch
import json
import os
from django.conf import settings


class Command(BaseCommand):
    """
    Azek:68184d52478abf06ec1a28ec
    Spirit_Airlines: 682f00def21b9fca8e1d04fe
    Catalent: 68347651c88b3f7f9c69410a
    ChampionX_Corp: 6836dbf3caf74b95439aeeba
    Silicon_Motion: 682f252ef21b9fca8e1d0530
    United_States_Steel: 684054f02e2e5aa5468773db
    Celgene_Corporation:68412f11812d9ee0838c6fd4


    example command:
    python manage.py search_schema 682f00def21b9fca8e1d04fe

    """
    help = 'Search all schema categories for a given deal ID using schema_by_summary_sections.json'

    def add_arguments(self, parser):
        parser.add_argument('deal_id', type=str,
                            help='The deal ID to search for')

    def handle(self, *args, **options):
        try:
            deal_id = options['deal_id']

            # Path to your schema file
            schema_path = os.path.join(
                settings.BASE_DIR, 'schema_by_summary_sections.json')

            # Initialize the search service
            search_service = SchemaCategorySearch()

            # Load the schema file
            with open(schema_path, 'r') as f:
                schema = json.load(f)
                self.stdout.write(self.style.SUCCESS(
                    f'Successfully loaded schema from {schema_path}'))

                # Override the schema in the search service
                search_service._schema = schema  # Set the schema directly

            # Perform the search
            self.stdout.write(self.style.SUCCESS(
                f'Searching schema categories for deal ID: {deal_id}'))
            results = search_service.search_all_schema_categories(deal_id)

            # Print results in a readable format
            self.stdout.write(json.dumps(results, indent=2))

            self.stdout.write(self.style.SUCCESS(
                'Search completed successfully'))

        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(
                f'Schema file not found at {schema_path}'))
        except json.JSONDecodeError:
            self.stdout.write(self.style.ERROR(
                'Error parsing schema file - invalid JSON'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error occurred: {str(e)}'))
