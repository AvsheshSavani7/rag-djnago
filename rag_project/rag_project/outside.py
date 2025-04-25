from pymongo import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus


username = "ashishkachadiya"
password = quote_plus("TeqnoDux888")

uri = f"mongodb+srv://{username}:{password}@cluster0.gdz2e7v.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    print("Databases:", client.list_database_names())
    print("✅ Connection success")
except Exception as e:
    print("❌ Failed:", e)
