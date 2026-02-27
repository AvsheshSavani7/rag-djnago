from django.core.management.base import BaseCommand
from rss_feeds.services import RSSFeedService
import json


class Command(BaseCommand):
    help = 'Test RSS feed webhook functionality with sample data'

    def handle(self, *args, **options):
        # Test data from the user
        test_payload = {
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

        self.stdout.write(self.style.SUCCESS('🧪 Testing RSS Feed Webhook Functionality'))
        self.stdout.write('=' * 60)
        
        try:
            # Process the webhook payload
            result = RSSFeedService.process_webhook_payload(test_payload)
            
            if result['success']:
                self.stdout.write(
                    self.style.SUCCESS(f"✅ Webhook processed successfully!")
                )
                self.stdout.write(f"Feed ID: {result['feed_id']}")
                self.stdout.write(f"Feed Title: {result['feed_title']}")
                self.stdout.write(f"Items Created: {result['items_created']}")
                self.stdout.write(f"Total Items Received: {result['total_items_received']}")
                
                # Test retrieving the created data
                self.stdout.write('\n📊 Testing Data Retrieval:')
                self.stdout.write('-' * 40)
                
                # Get all feeds
                feeds = RSSFeedService.get_all_feeds()
                self.stdout.write(f"Total feeds in database: {len(feeds)}")
                
                # Get recent items
                recent_items = RSSFeedService.get_recent_feed_items(limit=10)
                self.stdout.write(f"Recent items in database: {len(recent_items)}")
                
                if feeds:
                    feed = feeds[0]
                    feed_items = RSSFeedService.get_feed_items(str(feed.id))
                    self.stdout.write(f"Items for feed '{feed.title}': {len(feed_items)}")
                
                self.stdout.write(
                    self.style.SUCCESS('\n🎉 All tests completed successfully!')
                )
                
            else:
                self.stdout.write(
                    self.style.ERROR(f"❌ Webhook processing failed: {result['error']}")
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Error during testing: {str(e)}")
            )
