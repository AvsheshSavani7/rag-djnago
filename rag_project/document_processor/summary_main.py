# =========================
# Setup and Imports
# =========================
import sys
import os
import json
import boto3
from dotenv import load_dotenv
from summary_engine import process_clause_config, write_docx_summary
from summary_engine import RUN_CONCISE_SUMMARIES, RUN_FULSOME_SUMMARIES
# Load environment variables
load_dotenv()

# Configure AWS S3
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)
S3_BUCKET = os.getenv('AWS_S3_BUCKET')


def read_config_from_s3(config_name):
    """Read a config file from S3"""
    try:
        # Get the object from S3
        response = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key=f"clause_configs/{config_name}.py"
        )
        content = response['Body'].read().decode('utf-8')

        # Create a temporary module namespace
        namespace = {}
        exec(content, namespace)

        # Find the first uppercase variable which should be our config dictionary
        config_dict = next((val for name, val in namespace.items()
                            if name.isupper() and isinstance(val, dict)), {})

        return config_dict
    except Exception as e:
        print(f"Error reading config from S3: {str(e)}")
        return {}


def get_config_files_from_s3():
    """List all config files in S3"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix='clause_configs/'
        )

        config_files = []
        for obj in response.get('Contents', []):
            filename = os.path.basename(obj['Key'])
            if filename.endswith('_config.py') and not filename.startswith('__'):
                config_files.append(filename[:-3])  # Remove .py extension
        return sorted(config_files)
    except Exception as e:
        print(f"Error listing configs from S3: {str(e)}")
        return []


# =========================
# Load Config and Schema
# =========================
# Get list of available configs
config_files = get_config_files_from_s3()
print("Available configs:", config_files)

# Load all configs from S3
CLAUSE_CONFIG = {}
for config_name in config_files:
    config_dict = read_config_from_s3(config_name)
    CLAUSE_CONFIG.update(config_dict)

# =========================
# Accept JSON filename from terminal
# =========================
if len(sys.argv) < 2:
    print("❌ Please provide a JSON file name as an argument.\nUsage: python summary_main.py <filename.json>")
    sys.exit(1)

json_filename = sys.argv[1]

if not os.path.exists(json_filename):
    print(f"❌ File not found: {json_filename}")
    sys.exit(1)

with open(json_filename, "r", encoding="utf-8") as f:
    EXAMPLE_SCHEMA_DATA = json.load(f)

print("Loaded clause configs:", list(CLAUSE_CONFIG.keys()))


def parse_rank(rank):

    return [int(part) for part in str(rank).split(".")]


# =========================
# Run Clause Evaluations
# =========================
summary_outputs = []

for clause_name, clause_config in CLAUSE_CONFIG.items():

    summary_type = clause_config.get("summary_type", "Concise")

    # 🚨 ADD THIS LINE TO SKIP UNKNOWN OR DISABLED TYPES

    if summary_type not in ("Concise", "Fulsome"):

        print(
            f"Skipping {clause_name} — summary_type '{summary_type}' not recognized.")

        continue

    if summary_type == "Concise" and not RUN_CONCISE_SUMMARIES:

        continue

    if summary_type == "Fulsome" and not RUN_FULSOME_SUMMARIES:

        continue

    print(f"\n→ Evaluating: {clause_name}")
    result = process_clause_config(clause_config, EXAMPLE_SCHEMA_DATA)

    if result["output"] and result["output"] != "No output generated.":
        # Skip concise summaries where view_prompt is False
        if (
            result.get("summary_type", "") == "Concise"
            and clause_config.get("view_prompt", True) is False
        ):
            print(f"Skipping {clause_name} (concise, view_prompt=False)")
            continue

        summary_outputs.append({
            "clause_name": clause_name,
            **result
        })

        print("=== CLAUSE SUMMARY OUTPUT ===")
        print(f"Clause: {clause_name}")
        if result.get("used_prompt"):
            print("Used Prompt:\n" + result["used_prompt"])
        print("Summary:\n" + result["output"])
        if result.get("references"):
            print("References:")
            for r in result["references"]:
                print("- " + r)
        else:
            print("References: [None found or resolved]")

# =========================
# Write to DOCX
# =========================
# summary_outputs_sorted = sorted(summary_outputs, key=lambda x: x['summary_rank'])

summary_outputs_sorted = sorted(
    summary_outputs, key=lambda x: parse_rank(x['summary_rank']))

write_docx_summary(
    summary_outputs_sorted,
    json_filename.replace(".json", "_summary.docx"),
    RUN_CONCISE_SUMMARIES,
    RUN_FULSOME_SUMMARIES
)
print("\n✅ DOCX summary written.")
