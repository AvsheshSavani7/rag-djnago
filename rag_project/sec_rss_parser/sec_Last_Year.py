import requests
import json
from datetime import datetime, timedelta


def pad_cik(cik):
    """Pad CIK to 10 digits."""
    return cik.zfill(10)


def print_filings(cik, start_date=None, form_types=None):
    """Fixed for recent=DICT structure. start_date: if None, defaults to 1 year before today. form_types: list of form codes to filter (e.g. ['10-K','10-Q']); if None, no filter."""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)
                      ).strftime('%Y-%m-%d')
    if form_types is None:
        form_types = []
    form_types_set = {f.upper().strip()
                      for f in form_types} if form_types else None

    cik_padded = pad_cik(cik)
    url = f'https://data.sec.gov/submissions/CIK{cik_padded}.json'
    headers = {
        'User-Agent': 'KaushalDevani kaushal.devani@example.com (Regulatory Analyst)'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

    recent_dict = data['filings']['recent']
    if not recent_dict or 'accessionNumber' not in recent_dict:
        return []

    accession_numbers = recent_dict['accessionNumber']
    filing_dates = recent_dict['filingDate']
    forms = recent_dict['form']
    num_filings = len(accession_numbers)
    cik_int = int(data['cik'])
    result = []

    for i in range(num_filings):
        filing_date = filing_dates[i]
        form = forms[i]
        if form_types_set and (not form or form.upper() not in form_types_set):
            continue
        if not filing_date or filing_date < start_date:
            continue
        accession = accession_numbers[i]

        primary_doc = 'N/A'
        for key in ['primaryDocument', 'filename1', 'primary_doc']:
            if key in recent_dict and i < len(recent_dict[key]) and recent_dict[key][i]:
                primary_doc = recent_dict[key][i]
                break

        clean_acc = accession.replace('-', '')
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{clean_acc}/{primary_doc}"

        result.append({
            'filing_date': filing_date,
            'form': form,
            'accession_number': accession,
            'primary_document': primary_doc,
            'url': doc_url,
        })

    return result


# Run it
if __name__ == "__main__":
    cik = "1853513"
    # start_date=None -> default 1 year before today; form_types=None or [] -> no form filter
    filings = print_filings(cik)
    print(f"Returned {filings} ")
    # Or with params: print_filings(cik, start_date='2024-06-01', form_types=['10-K', '10-Q'])
