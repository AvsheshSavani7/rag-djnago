import json
import os
from datetime import datetime

import pytz
from django.core.management.base import BaseCommand

from document_processor.services import SchemaCategorySearch
from document_processor.transform_json import simplify_json
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
    Anywhere :68d14ef8530f016f4a3af0c2




    example command:
    python manage.py search_schema 68f2346697173821e21c5a71
    python manage.py search_schema 68d14ef8530f016f4a3af0c2 anywhere

    """
    help = 'Search all schema categories for a given deal ID using schema_by_summary_sections.json'

    def add_arguments(self, parser):
        parser.add_argument('deal_id', type=str,
                            help='The deal ID to search for')
        parser.add_argument(
            '--output',
            '-o',
            type=str,
            help='Optional path to save search results as JSON'
        )

    def handle(self, *args, **options):
        try:
            deal_id = options['deal_id']
            output_path = options.get('output')

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

            # Determine output path
            if output_path:
                target_path = os.path.abspath(output_path)
            else:
                # Match default filename pattern used when saving schema results
                ist = pytz.timezone('Asia/Kolkata')
                timestamp = datetime.now(ist).strftime("%d-%m-%y_%I-%M_%p")
                filename = f"schema_results_{timestamp}.json"
                target_path = os.path.join(settings.BASE_DIR, filename)

            # Always simplify results before saving
            data_to_save = simplify_json(results)

            # Save results to the determined path
            with open(target_path, 'w', encoding='utf-8') as outfile:
                json.dump(data_to_save, outfile, indent=2, ensure_ascii=False)

            if output_path:
                self.stdout.write(self.style.SUCCESS(
                    f'Search results (simplified) saved to {target_path}'))
            else:
                self.stdout.write(self.style.SUCCESS(
                    f'Search results (simplified) saved to {target_path}'))

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
