import requests
import re
import os
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def download_and_clean_book(url, output_file_path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        text = response.text

        # Remove Gutenberg header/footer
        start = text.find("*** START OF THIS PROJECT GUTENBERG EBOOK")
        end = text.find("*** END OF THIS PROJECT GUTENBERG EBOOK")
        if start != -1 and end != -1:
            text = text[start:end].replace('*** START OF THIS PROJECT GUTENBERG EBOOK', '').strip()

        # Clean text: remove BOM and control chars, non-ASCII, extra spaces, etc.
        # Re-evaluating regex to ensure no null bytes are accidentally introduced by bad patterns or ranges.
        # Keep only printable ASCII characters and common whitespace/punctuation. 
        # This pattern aims to be safer by explicitly allowing common characters.
        text = re.sub(r'[^\x20-\x7E\s]+', ' ', text)  # Keep printable ASCII and whitespace
        text = re.sub(r'\s+', ' ', text)            # Collapse whitespace
        text = text.replace('\n', ' ') # Ensure newlines within text are treated as spaces

        # Sentence tokenization
        sentences = sent_tokenize(text)

        with open(output_file_path, 'a', encoding="utf-8") as f:
            for sentence in sentences:
                cleaned_sentence = sentence.strip()
                if cleaned_sentence:
                    f.write(cleaned_sentence + "\n")
        print(f"Successfully processed: {url}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading or processing {url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {url}: {e}")

def main():
    # List of Project Gutenberg URLs for various books
    # You can add more URLs here to expand the dataset
    gutenberg_urls = [
        "https://www.gutenberg.org/files/11/11-0.txt",  # Alice in Wonderland
        "https://www.gutenberg.org/files/1342/1342-0.txt", # Pride and Prejudice
        "https://www.gutenberg.org/files/2701/2701-0.txt", # Moby Dick
        "https://www.gutenberg.org/files/2542/2542-0.txt", # A Christmas Carol
        "https://www.gutenberg.org/files/74/74-0.txt", # The Adventures of Tom Sawyer
        # Add more as desired
    ]

    DATA_DIR = os.path.dirname(__file__)
    CORPUS_PATH = os.path.join(DATA_DIR, 'corpus.txt')

    # Clear existing corpus.txt before appending new data
    if os.path.exists(CORPUS_PATH):
        os.remove(CORPUS_PATH)
        print(f"Removed existing {CORPUS_PATH}")

    print(f"Starting dataset collection. Output will be appended to {CORPUS_PATH}")
    for url in gutenberg_urls:
        download_and_clean_book(url, CORPUS_PATH)
    
    print("Dataset collection complete.")

if __name__ == "__main__":
    main() 