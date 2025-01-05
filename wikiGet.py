import requests
import json
import time
import argparse
import pypandoc

# Global Variables
WIKI_URL = "https://wiki.moltenaether.com/db/api.php"
API_KEY = None  # Optional: Provide an API key here if required (e.g., "your_api_key")


def fetch_category_pages(category):
    """
    Fetch all pages in a given category.
    """
    pages = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "format": "json",
            "cmcontinue": cmcontinue,
        }
        if API_KEY:
            params["apikey"] = API_KEY

        response = requests.get(WIKI_URL, params=params).json()
        pages.extend(member["title"] for member in response["query"]["categorymembers"])

        cmcontinue = response.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    return pages


def fetch_page_links(title):
    """
    Fetch all linked pages on a given page.
    """
    links = []
    plcontinue = None

    while True:
        params = {
            "action": "query",
            "titles": title,
            "prop": "links",
            "format": "json",
            "plcontinue": plcontinue,
        }
        if API_KEY:
            params["apikey"] = API_KEY

        response = requests.get(WIKI_URL, params=params).json()
        pages = response["query"]["pages"]
        for page_id, page_info in pages.items():
            if "links" in page_info:
                links.extend(link["title"] for link in page_info["links"])

        plcontinue = response.get("continue", {}).get("plcontinue")
        if not plcontinue:
            break

    return links


def fetch_page_content(title):
    """
    Fetch the wikitext content of a given page.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "format": "json",
    }
    if API_KEY:
        params["apikey"] = API_KEY

    response = requests.get(WIKI_URL, params=params).json()
    pages = response["query"]["pages"]
    for page_id, page_info in pages.items():
        if "revisions" in page_info:
            return page_info["revisions"][0]["*"]

    return None


def crawl_wiki(start_category):
    """
    Crawl the wiki starting from a given category.
    """
    visited = set()
    queue = fetch_category_pages(start_category)
    result = []

    while queue:
        title = queue.pop(0)
        if title in visited:
            continue
        visited.add(title)

        print(f"Processing: {title}")
        content = fetch_page_content(title)
        if content:
            result.append({"title": title, "content": content})

        # Get linked pages and add them to the queue
        links = fetch_page_links(title)
        queue.extend(link for link in links if link not in visited)

        # Respect API limits
        time.sleep(0.5)

    return result


def save_to_json(data, filename):
    """
    Save data to a JSON file.
    """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def convert_to_markdown(data):
    """
    Convert the content of each article in the dataset from wikitext to Markdown.

    Args:
        data (list): A list of dictionaries where each dictionary contains 'title' and 'content'.

    Returns:
        list: The same list with the 'content' converted to Markdown.
    """
    for article in data:
        try:
            article['content'] = pypandoc.convert_text(article['content'], 'markdown', format='mediawiki')
        except Exception as e:
            print(f"Error converting article '{article['title']}': {e}")
            article['content'] = f"Conversion failed: {e}"
    return data


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Wiki Crawler Script")
    parser.add_argument(
        "-c",
        "--category",
        required=True,
        help="The starting category to crawl (e.g., 'Example Category').",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="The name of the output JSON file (e.g., 'output.json').",
    )

    args = parser.parse_args()

    # Crawl the wiki and save results
    data = crawl_wiki(args.category)
    save_to_json(data, args.output)

    print(f"Data saved to {args.output}")


if __name__ == "__main__":
    main()
