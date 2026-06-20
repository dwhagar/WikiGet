# WikiGet for NotebookLM

This project contains `wikiGet.py`, a Python script designed to download all content from a MediaWiki site and save it as a structured, multi-file collection of Markdown documents, optimized for analysis with Google's NotebookLM.

## Features

-   **Advanced NLP Keyword Extraction**: Uses `spaCy` and `NLTK` to perform high-quality linguistic analysis, including lemmatization, part-of-speech analysis, named entity recognition, and noun chunking.
-   **Intelligent Indexing**: Automatically generates three powerful index files: `_Master_Index.md`, `_Category_Index.md`, and `_Keyword_Index.md`.
-   **Smart File Bundling**: To stay within NotebookLM's file limits, the script intelligently bundles small pages into a single file (`_Small_Pages.md`).
-   **Configuration Driven**: All settings are managed via a `config.json` file.

## Installation and Setup

This project features a fully automated setup process.

Navigate to the project's root directory in your terminal and run:
```bash
pip install .
```
This single command will:
1.  Install the `wikiget` script.
2.  Install all required Python libraries (`spacy`, `nltk`, `requests`, etc.).
3.  Automatically download the necessary `nltk` stopwords and `spaCy` language models.

There are no further manual setup steps required.

## Output Structure

The script generates an output directory containing:

1.  **`_Master_Index.md`**: A master list of all pages with links to their content and metadata.
2.  **`_Category_Index.md`**: An index grouping pages by category.
3.  **`_Keyword_Index.md`**: An index of all extracted keywords and phrases, linking back to the pages where they appear.
4.  **Individual Page Files**: A separate `.md` file for each "large" page.
5.  **`_Small_Pages.md`**: A single file containing all "small" pages, with each page marked by a clear header and anchor.

## Usage

Once installed, you can run the script from any directory using the `wikiget` command:

```bash
wikiget [OPTIONS]
```

### Options

-   `-c, --config FILE`: Path to a JSON configuration file (default: `config.json`).
-   `-o, --output DIRECTORY`: Output directory for the generated files (overrides config).
-   `--workers N`: Number of concurrent workers (overrides config).
-   `--test-run`: Process only the first 20 pages.
-   `-v, --verbose`: Enable detailed logging.

## Configuration

The script is controlled by a `config.json` file. It is recommended to copy `config.json.example` to `config.json` and modify it.

```json
{
  "wikiGet": {
    "wiki_url": "https://your-wiki.com/w/api.php",
    "api_key": "",
    "category_blacklist": [],
    "page_blacklist": [],
    "output_directory": "output",
    "workers": 8,
    "test_run": false,
    "verbose": false,
    "small_page_threshold": 20000,
    "key_phrase_threshold": 20000
  }
}
```

### Configuration Fields

-   `small_page_threshold`: The character count below which a page is considered "small" and will be bundled.
-   `key_phrase_threshold`: The character count above which the script will extract multi-word key phrases in addition to single keywords.
