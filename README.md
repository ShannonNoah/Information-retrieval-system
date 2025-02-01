# CSI4107 - Assignment 1: Information Retrieval System (GROUP 45)

This repository contains our groups implementation of an Information Retrieval (IR) system based on the vector space model for the Scifact dataset.

---

## **Setup Instructions**

### 1. Install Required Libraries
Ensure Python 3.8 or higher is installed on your system. Install the required Python libraries by running:
```bash
pip install nltk beautifulsoup4
```

### 2. Download NLTK Data
After installing the libraries, download the necessary NLTK datasets:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
These commands prepare the environment for text preprocessing, including stopwords removal and tokenization.

## Project Structure

- **`preprocess.py`**: Script to clean and prepare text data for indexing.
- **`index.py`**: Constructs an inverted index from the preprocessed data.
- **`vocabulary.txt`**: Stores all unique tokens extracted from the text.
- **`dataset`**: Contains the `corpus.jsonl`, `queries.jsonl`, `train.csv`, and `test.csv`.
- **`inverted_index.json`**: The final outputted inverted index.

## Usage

### Running the Preprocessing Script
Navigate to the project directory and run:
```bash
python preprocess.py
```
This processes `dataset/corpus.jsonl`, handling tokenization, stopwords removal, and stemming to produce `vocabulary.txt`.

### Building the Inverted Index
To build the inverted index, run:
```bash
python index.py
```
This script utilizes the vocabulary and documents from `dataset/corpus.jsonl` to create an index mapping tokens to their document locations and occurrence frequencies.

### Viewing the Output
The created inverted index is saved in `inverted_index.json`. Open this file with any JSON-supporting text editor to examine the structured index.

## Features

- **Text Preprocessing**: Involves tokenization, removal of stopwords, and stemming.
- **Inverted Index Construction**: Efficiently maps tokens to document IDs and counts occurrences.
- **JSON Data Management**: Manages the reading and processing of JSON formatted data.

## Contributions

Shabrina Sharmin
Shannon Noah
Tina Trinh
