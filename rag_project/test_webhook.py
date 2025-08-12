#!/usr/bin/env python
"""
Test script for RSS feed webhook functionality
"""
import json
import requests
from datetime import datetime

# Test data from the user
TEST_WEBHOOK_DATA = {
    "id": "evt_hPsGadJ3yy7kWUC7",
    "type": "feed_update",
    "feed": {
        "id": "DLkVWM0xLG90tUiD",
        "title": "All Acquisitions, Mergers and Takeovers News and Press Releases from PR Newswire",
        "source_url": "https://www.prnewswire.com/news-releases/financial-services-latest-news/acquisitions-mergers-and-takeovers-list/",
        "rss_feed_url": "https://rss.app/feeds/DLkVWM0xLG90tUiD.xml",
        "description": "Acquisitions, Mergers and Takeovers",
        "icon": "https://www.prnewswire.com/content/dam/prnewswire/icons/2019-Q4-PRN-Icon-32-32.png"
    },
    "data": {
        "items_new": [
            {
                "url": "https://www.prnewswire.com/news-releases/cango-inc-acquisisce-un-impianto-di-mining-di-bitcoin-da-50-mw-in-georgia-gettando-le-basi-per-la-futura-strategia-energetica-302526896.html",
                "title": "Cango Inc. acquisisce un impianto di mining di bitcoin da 50 MW in Georgia, gettando le basi per la futura strategia energetica",
                "description_text": "/PRNewswire/ -- Cango Inc. (NYSE: CANG) (\"Cango\" o la \"Società\"), ha annunciato oggi l'acquisizione di un impianto di mining da 50 MW completamente operativo...",
                "thumbnail": "https://mma.prnewswire.com/media/2675436/CANG_LOGO_Logo.jpg?p=twitter",
                "date_published": "2025-08-11T22:13:00.000Z",
                "authors": [{"name": "Cango Inc."}]
            },
            {
                "url": "https://www.prnewswire.com/news-releases/clarimed-inc-acquires-we-are-human-expanding-growth-into-southeast-market-302527089.html",
                "title": "ClariMed Inc. Acquires We Are Human, Expanding Growth into Southeast Market",
                "description_text": "/PRNewswire/ -- ClariMed, Inc., a global leader in human-centered medical device development and regulatory services, today announced the strategic acquisition...",
                "thumbnail": "https://mma.prnewswire.com/media/2748400/We_Are_Human.jpg?p=twitter",
                "date_published": "2025-08-12T07:00:00.000Z",
                "authors": [{"name": "ClariMed Inc."}]
            },
            {
                "url": "https://www.prnewswire.com/news-releases/clarimed-inc-acquires-we-are-human-expanding-growth-into-southeast-market-302527106.html",
                "title": "ClariMed Inc. Acquires We Are Human, Expanding Growth into Southeast Market",
                "description_text": "/PRNewswire/ -- ClariMed, Inc., a global leader in human-centered medical device development and regulatory services, today announced the strategic acquisition...",
                "thumbnail": "https://mma.prnewswire.com/media/2748400/We_Are_Human.jpg?p=twitter",
                "date_published": "2025-08-12T07:00:00.000Z",
                "authors": [{"name": "ClariMed Inc."}]
            }
        ],
        "items_changed": []
    }
}


def test_webhook_endpoint():
    """Test the webhook endpoint with the provided data"""

    # Webhook endpoint URL (adjust as needed)
    webhook_url = "http://localhost:8000/api/rss/webhook/"

    print("🧪 Testing RSS Feed Webhook Endpoint")
    print("=" * 50)
    print(f"URL: {webhook_url}")
    print(f"Data: {json.dumps(TEST_WEBHOOK_DATA, indent=2)}")
    print("-" * 50)

    try:
        # Send POST request to webhook endpoint
        response = requests.post(
            webhook_url,
            json=TEST_WEBHOOK_DATA,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            print("✅ Webhook test successful!")
        else:
            print("❌ Webhook test failed!")

    except requests.exceptions.ConnectionError:
        print(
            "❌ Connection error: Make sure the Django server is running on localhost:8000")
    except requests.exceptions.Timeout:
        print("❌ Request timeout")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def test_api_endpoints():
    """Test the API endpoints after webhook processing"""

    base_url = "http://localhost:8000/api/rss"

    print("\n🧪 Testing API Endpoints")
    print("=" * 50)

    endpoints = [
        ("/feeds/", "Get all feeds"),
        ("/items/", "Get recent feed items"),
    ]

    for endpoint, description in endpoints:
        try:
            url = base_url + endpoint
            print(f"\n{description}: {url}")

            response = requests.get(url, timeout=10)
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"Count: {data.get('count', 'N/A')}")
                print("✅ Success")
            else:
                print("❌ Failed")

        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    # Test webhook endpoint
    test_webhook_endpoint()

    # Test API endpoints
    test_api_endpoints()

    print("\n�� Test completed!")
