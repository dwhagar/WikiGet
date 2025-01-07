import requests
import json
import time
import argparse
import pypandoc
import hashlib

# Global Variables
WIKI_URL = "https://wiki.moltenaether.com/w/api.php"
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


def fetch_page_links(title, visited):
    """
    Fetch all linked pages on a given page, following redirects if necessary.
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
            if "redirects" in page_info:  # Check if it's a redirect
                redirect_title = page_info["redirects"][0]["title"]
                print(f"Redirected from {title} to {redirect_title}")
                if redirect_title not in visited:  # Only follow the redirect if it hasn't been visited yet
                    visited.add(redirect_title)  # Mark the redirected page as visited
                    return fetch_page_links(redirect_title, visited)  # Recursively fetch the target page
                else:
                    return []  # If we've already visited the redirected page, return empty links

            if "links" in page_info:
                links.extend(link["title"] for link in page_info["links"])

        plcontinue = response.get("continue", {}).get("plcontinue")
        if not plcontinue:
            break

    return links


def fetch_page_content(title, visited):
    """
    Fetch the wikitext content and categories of a given page, following redirects if necessary.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions|links|categories",  # Fetch content, links (to detect redirect), and categories
        "rvprop": "content",
        "format": "json",
    }
    if API_KEY:
        params["apikey"] = API_KEY

    response = requests.get(WIKI_URL, params=params).json()
    pages = response["query"]["pages"]

    categories = []  # List to store categories

    for page_id, page_info in pages.items():
        if "revisions" in page_info:
            content = page_info["revisions"][0]["*"]  # Page content (wikitext)

            # Check for redirect (i.e., #REDIRECT at the start of the page)
            if content.startswith("#REDIRECT"):
                # Extract the redirect target from the content
                redirect_target = content.split("\n")[0].replace("#REDIRECT", "").strip()
                if redirect_target[0] == ":":
                    redirect_target = redirect_target[1:]
                if redirect_target[0:2] == "[[":
                    redirect_target = redirect_target.strip("[[")
                if redirect_target[-2:] =="]]":
                    redirect_target = redirect_target.strip("]]")
                if redirect_target[0] == ":":
                    redirect_target = redirect_target[1:]

                redirect_target = redirect_target.split("#")[0]

                # Check if the target is a valid page title and it's not visited
                if redirect_target not in visited:
                    visited.add(redirect_target)  # Mark the redirected page as visited
                    print(f"Redirected from {title} to {redirect_target}")
                    return fetch_page_content(redirect_target, visited)  # Recursively fetch the target page
                else:
                    print(f"Skipping already visited redirect: {redirect_target}")
                    return None, categories  # If redirected page is already visited, skip it

            # Collect categories if they exist
            if "categories" in page_info:
                categories = [category["title"] for category in page_info["categories"]]

            return content, categories  # Return both content and categories

    return None, categories  # Return None if page has no content


def get_content_hash(content):
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def crawl_wiki(queue, visited, visited_content, result):
    base_url = "https://wiki.moltenaether.com/db/"

    while queue:
        title = queue.pop(0)
        if title in visited:
            continue
        visited.add(title)

        # Follow links and get content, handling redirects if necessary
        links = fetch_page_links(title, visited)

        print(f"Processing: {title}")
        # Define a list of banned categories
        banned_categories = ["Abandoned"]

        # Fetch content and categories
        content, categories = fetch_page_content(title, visited)

        # Check if content exists and if any categories are banned
        if content and not any(banned_category.lower()
                               in category.lower() for category
                               in categories for banned_category
                               in banned_categories):
            content_hash = get_content_hash(content)
            if content_hash in visited_content:
                continue  # Skip if content is already visited

            visited_content.add(content_hash)

            # Construct the URL for the article
            article_url = base_url + title.replace(" ", "_")

            # Add the URL and categories field to the result
            result.append({
                "title": title,
                "content": content,
                "links": links,
                "url": article_url,
                "categories": categories  # Add categories field
            })

        # Get linked pages and add them to the queue
        queue.extend(link for link in links if link not in visited)

        # Respect API limits
        time.sleep(0.5)

    return result


import json


def save_to_json(data, filename: str) -> None:
    """
    Save data to a JSON file by first converting to a string.
    """
    # Convert the data to a JSON-formatted string
    json_data = json.dumps(data, ensure_ascii=False, indent=4)

    # Write the string to the file
    with open(filename, "w", encoding="utf-8") as output_file:
        output_file.write(json_data)


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

    # Initialize variables
    visited = set()
    visited_content = set()
    result = []

    # Get list of pages in the category
    queue = fetch_category_pages(args.category)

    # Crawl the wiki and save results
    data = crawl_wiki(queue, visited, visited_content, result)
    save_to_json(data, args.output)

    print(f"Data saved to {args.output}")


if __name__ == "__main__":
    main()
