from sec_api import QueryApi
import os
import requests

# Replace this with your API key
API_KEY = "11fb1dcbce38b9e0bfde4b4cb51b1dc2fee16d13c2da36bbbed172542d1aaacf"
query_api = QueryApi(api_key=API_KEY)

# Create a folder to store raw files
os.makedirs("tesla_sec_filings", exist_ok=True)

# Define your query: Tesla's 10-K and 10-Q
query = {
    "query": {
        "query_string": {
            "query": "ticker:TSLA AND (formType:\"10-K\" OR formType:\"10-Q\")"
        }
    },
    "from": "0",
    "size": "10",  # You can increase this to get more filings
    "sort": [{ "filedAt": { "order": "desc" } }]
}

results = query_api.get_filings(query)

for filing in results["filings"]:
    url = filing["linkToTxt"]
    form_type = filing["formType"]
    filed_date = filing["filedAt"][:10]
    
    # Clean filename
    fname = f"tesla_sec_filings/TESLA_{form_type}_{filed_date}.txt"
    print(f"Downloading {form_type} from {filed_date}...")

    r = requests.get(url)
    with open(fname, "w", encoding="utf-8") as f:
        f.write(r.text)

print("✅ Download complete.")
