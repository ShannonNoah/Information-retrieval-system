import json
import os


def load_vocabulary(vocab_file):
    """Load vocabulary from a file into a set for quick lookup."""

    try:
        with open(vocab_file, "r", encoding="utf-8") as file:
            vocabulary = set(line.strip() for line in file)

        print(f"Loaded {len(vocabulary)} words from vocabulary.")
        return vocabulary

    except FileNotFoundError:
        print(f"Error: The vocabulary file '{vocab_file}' was not found.")
        return None


def inverted_index(corpus_file, vocab):
    """Build an inverted index from the corpus using the specified vocabulary."""

    if not os.path.exists(corpus_file):
        print(f"Error: The file {corpus_file} does not exist.")
        return {}

    index = {}

    try:
        with open(corpus_file, "r", encoding="utf-8") as file:

            for line in file:
                doc = json.loads(line.strip())
                doc_id = doc["_id"]  # Document ID from JSON
                tokens = doc["text"].split()  # Tokenize the text

                for token in tokens:
                    if token in vocab:
                        if token not in index:
                            index[token] = {doc_id: 1}
                        else:
                            index[token][doc_id] = index[token].get(doc_id, 0) + 1

    except json.JSONDecodeError as e:
        print(f"JSON decoding error in {corpus_file}: {e}")

    except Exception as e:
        print(f"An error occurred while processing {corpus_file}: {e}")

    return index


def save_index(index, output_file):
    """Save the inverted index to a JSON file."""

    try:
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(index, file)
        print(f"Saved index to {output_file}.")

    except IOError as e:
        print(f"Could not write to file {output_file}: {e}")


if __name__ == "__main__":
    vocabulary = load_vocabulary("vocabulary.txt")

    if vocabulary:
        index = inverted_index("dataset/corpus.jsonl", vocabulary)

        if index:
            save_index(index, "inverted_index.json")
