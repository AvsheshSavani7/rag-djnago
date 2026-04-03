#Old flow scripts are below
#---------------------------------------#

covenant_scraping.py
    Input 
        urls.txt(read)


    Output files:
        openai_response_{accession}_full.json (New created)
        openai_response_{accession}_covenants.json (New created)
        openai_response_{accession}_individual_clauses.json (New created)

Stage 6 - 6_classify_new_deal.py
    Input 
        openai_response_{deal_id}_individual_clauses.json
        Covenant_Embeddings_v1/embeddings_cache/covenant_clustering_*.json (read)
        Covenant_Embeddings_v1/embeddings_cache/covenant_cluster_analysis_*.json (read)

    Output files:
        new_deal_reports/deal_classification_{deal_id}_{timestamp}.json (New created)
        new_deal_reports/deal_summary_{deal_id}_{timestamp}.csv (New created)

Stage 7 - 7_assess_new_deal.py
    Input 
        new_deal_reports/deal_classification_{deal_id}_{timestamp}.json (read)

    Output files:
        new_deal_reports/deal_assessment_{deal_id}_{timestamp}.json(New created)

Stage 8 - 8_compare_to_benchmark.py
    Input 
        new_deal_reports/deal_assessment_{deal_id}_{timestamp}.json (read)
        Covenant_Embeddings_v1/embeddings_cache/covenant_assessment_summary_*.json (read)
        Covenant_Embeddings_v1/embeddings_cache/covenant_risk_assessment_*.json (read, optional)

    Output files:
        new_deal_reports/benchmark_comparison_{deal_id}_{timestamp}.json(New created)
        new_deal_reports/benchmark_summary_{deal_id}_{timestamp}.csv(New created)

Stage 9 - 9_specific_provision_checks.py
    Input 
        new_deal_reports/openai_response_{deal_id}_individual_clauses.json (read)

    Output files:
        new_deal_reports/specific_provisions_{deal_id}_{timestamp}.json (new created)


   

Copy latest file from new_deal_reports folder into input folder(input/{deal_id}/)
    deal_classification_*.json
    deal_assessment_*.json
    benchmark_comparison_*.json
    specific_provisions_*.json

Stage 10 - 10_generate_dashboard.py
    Input 
        input/{deal_id}/
            deal_classification_*.json from Stage 6 (read)
            deal_assessment_{deal_id}_*.json from Stage 7 (read)
            benchmark_comparison_{deal_id}_*.json from Stage 8 (read, optional)
            specific_provisions_{deal_id}_*.json from Stage 9 (read, optional)

    Output files:
        dashboard_output/covenant_dashboard_{deal_id}_{timestamp}.html (New created)


#---------------------------------------#
#New Flow input output are below:
#---------------------------------------#

1)modify scripts:
    modify covenant_scraping.py to save s3 urls instead of local files.
    modify 6_classify_new_deal.py, 7_assess_new_deal.py, 8_compare_to_benchmark.py, 9_specific_provision_checks.py to save s3 urls instead of local files.
    modify 10_generate_dashboard.py to use s3 urls instead of local files.
    modify covenant_processor.py to use s3 urls instead of local files.
    remove old flow scripts.
    refere env file termination_pipeline.py for new flow scripts.
    refere main structure of termination_pipeline.py for new flow scripts.
change path where needed.

New collection name: covenant_analysis

schema:
    deal_id: string (required) from deals collection[not from document extraction]
    sec_url: string (required)
    accession_number: string (required)
    full_json: string (s3 json url)
    covenants_json: string (s3 json url)
    individual_clauses_json: string (s3 json url)
    classification_json: string (s3 json url)
    summary_csv: string (s3 csv url)
    assessment_json: string (s3 json url)
    benchmark_comparison_json: string (s3 json url)
    benchmark_summary_csv: string (s3 csv url)
    specific_provisions_json: string (s3 json url)
    dashboard_html: string (s3 html url)
    created_at: datetime
    updated_at: datetime

In S3:
    covenant_analysis/accession_number/full_json.json
    covenant_analysis/accession_number/covenants_json.json
    covenant_analysis/accession_number/individual_clauses_json.json
    covenant_analysis/accession_number/classification_json.json
    covenant_analysis/accession_number/summary_csv.csv
    covenant_analysis/accession_number/assessment_json.json
    covenant_analysis/accession_number/benchmark_comparison_json.json
    covenant_analysis/accession_number/benchmark_summary_csv.csv
    covenant_analysis/accession_number/specific_provisions_json.json
    covenant_analysis/accession_number/dashboard_html.html

Note:
deal_id:in mongoDB deal_id is from deals collection so in script we extract deal id and add in json but that deal id is different. so dont confuse with that deal id.so in collection record and in s3 we use deal_id from deals collection.other deal_id in jsons are leave as it is.
Now there is no new_deal_reports folder. so we use s3 urls for classification, assessment, provision_checks and dashboard.
also now no data.covenant.input folder. so we use s3 urls for full_json, covenants_json, individual_clauses_json.