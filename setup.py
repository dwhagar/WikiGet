import setuptools
from setuptools.command.install import install
import sys

class PostInstallCommand(install):
    """Post-installation for NLTK and spaCy model downloads."""
    def run(self):
        # Run the standard install process
        install.run(self)
        
        # Now that the install is complete, the libraries are in the environment.
        # We can import them directly and run their download commands.
        try:
            print("--- Running post-install commands ---")
            print("Importing nltk and downloading 'stopwords' model...")
            import nltk
            nltk.download('stopwords')
            print("NLTK 'stopwords' model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading NLTK model: {e}", file=sys.stderr)
            print("Please try running 'python -c \"import nltk; nltk.download(\'stopwords\')\"' manually.", file=sys.stderr)

        try:
            print("Importing spacy and downloading 'en_core_web_sm' model...")
            import spacy.cli
            spacy.cli.download('en_core_web_sm')
            print("spaCy 'en_core_web_sm' model downloaded successfully.")
            print("--- Post-install commands complete ---")
        except Exception as e:
            print(f"Error downloading spaCy model: {e}", file=sys.stderr)
            print("Please try running 'python -m spacy download en_core_web_sm' manually.", file=sys.stderr)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wiki-get-notebooklm",
    version="1.0.0",
    author="WikiGet User",
    author_email="user@example.com",
    description="A tool to download a MediaWiki site into a structured, multi-file format for analysis with NotebookLM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/wiki-get",  # Replace with your actual URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Assuming MIT, change if needed
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
    install_requires=[
        "requests",
        "beautifulsoup4",
        "markdownify",
        "rich",
        "spacy",
        "nltk",
    ],
    entry_points={
        'console_scripts': [
            'wikiget=wikiGet:main',
        ],
    },
    cmdclass={
        'install': PostInstallCommand,
    },
)
