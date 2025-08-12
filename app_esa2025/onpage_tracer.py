import requests
import re
from urllib.parse import urlparse

SPIDER_API_KEY = open("secrets/spider").read().strip()

headers = {
    "Authorization": f"Bearer {SPIDER_API_KEY}",
    "Content-Type": "application/json",
}

# Set parameters for initial crawl
initial_url = "https://events.rdmobile.com/Sessions/Index/19095"
domain_to_follow = "events.rdmobile.com"
phrase_to_match = "ESA 2025"
limit = 20  # Adjust as needed

json_data = {
    "limit": limit,
    "return_format": "markdown",
    "url": initial_url,
    "follow_internal_links": True,  # Ensure internal links are crawled
    "smart": True,  # Render dynamic JavaScript content if necessary
}

# Perform the initial crawl
response = requests.post(
    "https://api.spider.cloud/crawl", headers=headers, json=json_data
)

results = response.json()

# Store matched pages
matched_pages = []

# Check if crawl was successful
if "items" in results:
    for item in results["items"]:
        url = item.get("url", "")
        content = item.get("body", "")

        # Ensure URL is within the specified domain
        parsed_url = urlparse(url)
        if parsed_url.netloc.endswith(domain_to_follow):
            if phrase_to_match.lower() in content.lower():
                matched_pages.append({"url": url, "content": content})

    # Display matched pages
    for page in matched_pages:
        print(f"Matched URL: {page['url']}\n")
        print(
            f"Content:\n{page['content'][:500]}...\n"
        )  # print first 500 chars
        print("-" * 80)
else:
    print("Error or no items returned:", results)
