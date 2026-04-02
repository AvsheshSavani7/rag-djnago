#Old flow scripts are below
#---------------------------------------#

1)termination_pipeline.py:
use all below file (scraping, stage 6, stage 7, stage 9)
termination_scraping.py
    Input urls.txt(read)

    agreement path
        termination_response_*_full.json(new created)
        termination_response_*_triggers.json(new created)
        termination_response_*_fees.json(new created)
        termination_response_*_triggers_raw.json
    8-K / press release path
        termination_response_*_triggers_8k.json(new created)
        termination_response_*_fees_8k.json(new created)
    If warnings/errors exist
        termination_response_{accession}.txt(new created)


Stage 6 — 6_classify_new_deal_termination.py
   Input termination_response_*_triggers.json
   

   embeddings_cache/termination_clustering_*.json(read)
   embeddings_cache/termination_cluster_analysis_*.json(read)
   final_results/termination_benchmark_*.json(read)


   new_deal_reports/termination_classification_{deal_id}_{timestamp}.json(new created)
   new_deal_reports/termination_summary_{deal_id}_{timestamp}.csv(new created)


Stage 7 — 7_assess_new_deal_termination.py
    Input termination_classification_{deal_id}_{timestamp}.json(read)

    final_results/termination_benchmark_*.json(read)

    new_deal_reports/termination_assessment_{deal_id}_{timestamp}.json(new created)

Stage 9 — 9_specific_provision_checks_termination.py
    Input termination_response_*_fees.json(read)
    termination_response_{accession}_triggers.json (read, optional)

    new_deal_reports/termination_provision_checks_{deal_id}_{timestamp}.json(new created)



Copy latest Stage 6 / 7 / 9 reports into input folder

Copy latest trigger/fee source files from all source types(8-K, agreement)




Final Input json like:
    termination_classification_69c...json
    termination_assessment_69c...json
    termination_provision_checks_69c...json
    termination_response_..._triggers.json
    termination_response_..._triggers_8k.json
    termination_response_..._fees.json
    termination_response_..._fees_8k.json

2)termination_processor.py:
for generate html dashboard from s3 urls.


#---------------------------------------#
#New Flow input output are below:
#---------------------------------------#

1)modify scripts:
    modify termination_pipeline.py to use s3 urls instead of local files.also sorting logic not needed because we have specific files for each stage.also always we use full extraction flow not caching.
    modify termination_scraping.py to save s3 urls instead of local files.
    modify 6_classify_new_deal_termination.py, 7_assess_new_deal_termination.py, 9_specific_provision_checks_termination.py to save s3 urls instead of local files.
    modify 10_generate_dashboard_termination.py to use s3 urls instead of local files.
    modify termination_processor.py to use s3 urls instead of local files.
change path where needed.

New collection name: termination_analysis
schema:
    deal_id: string (required) from deals collection[note from document extraction]
    sec_url: string (required)
    accession_number: string (required)
    doc_type: string (required) ("8-K" or "99.1" or "2.1")
    full_json: string (s3 json url)
    triggers_json: string (s3 json url)
    fees_json: string (s3 json url)
    triggers_raw_json: string (s3 json url)
    triggers_8k_json: string (s3 json url)
    fees_8k_json: string (s3 json url)

    classification_json: string (s3 json url)
    summary_csv: string (s3 csv url)

    assessment_json: string (s3 json url)

    provision_checks_json: string (s3 json url)

    dashboard_html: string (s3 html url)

    created_at: datetime
    updated_at: datetime



In S3:
    termination_analysis/accession_number/doc_type/full_json.json
    termination_analysis/accession_number/doc_type/triggers_json.json
    termination_analysis/accession_number/doc_type/fees_json.json
    termination_analysis/accession_number/doc_type/triggers_raw_json.json
    termination_analysis/accession_number/doc_type/triggers_8k_json.json
    termination_analysis/accession_number/doc_type/fees_8k_json.json
    termination_analysis/accession_number/doc_type/classification_json.json
    termination_analysis/accession_number/doc_type/summary_csv.csv
    termination_analysis/accession_number/doc_type/assessment_json.json
    termination_analysis/accession_number/doc_type/provision_checks_json.json
    termination_analysis/accession_number/doc_type/dashboard_html.html


Note:
deal_id:in mongoDB deal_id is from deals collection so in script we extract deal id and add in json but that deal id is different. so dont confuse with that deal id.so in collection record and in s3 we use deal_id from deals collection.other deal_id in jsons are leave as it is.
Now there is no new_deal_reports folder. so we use s3 urls for classification, assessment, provision_checks and dashboard.
also now no data.termination.input folder. so we use s3 urls for full_json, triggers_json, fees_json, triggers_raw_json, triggers_8k_json, fees_8k_json.