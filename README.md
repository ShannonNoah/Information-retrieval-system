# CSI4107 - Assignment 1: Information Retrieval System (GROUP 45)

This repository contains our groups implementation of an Information Retrieval (IR) system based on the vector space model for the Scifact dataset.

---

## **Setup Instructions**

### 1. Install Required Libraries

Ensure Python 3.8 or higher is installed on your system. Install the required Python libraries by running:

```bash
pip install nltk beautifulsoup4 itertools tabulate pandas uuid
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
- **`retr_rank.py`** : Constructs the document ranking based on cosine similarity of the query.
- **`Results.txt`** : The output file containing the ranking of the top 100 documents for the query based on relevance.

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

### Ranking the document

To build the raking, run:

```bash
python retr_rank.py
```

The script uses the modules index.py and preprocess.py as well as the output files `inverted_index.json` and `vocabulary.txt` to create the ranking based on the `dataset/queries.jsonl` file.

### Viewing the Output

The created inverted index is saved in `inverted_index.json`. Open this file with any JSON-supporting text editor to examine the structured index.

The ranking of the documents is stored in `Results.txt`. Open this file in a text editor to view the ranking of the documents for the queries.

## Features

- **Text Preprocessing**: Involves tokenization, removal of stopwords, and stemming.
- **Inverted Index Construction**: Efficiently maps tokens to document IDs and counts occurrences.
- **JSON Data Management**: Manages the reading and processing of JSON formatted data.
- **IR Vector space model**: Uses the cosine similarity to rank the documents for queries.

## Contributions

Shabrina Sharmin

Shannon Noah

Tina Trinh

## Results of trec_eval

Steps to run:

1. Open the evaluate.ipynb file in your Jupyter Notebook environment.
2. Upload the following files to the notebook: Results.txt, test.tsv
3. Run the cells.

Below are the key results:
runid all 5d7d54f3-bbcd-4a7b-ae8d-78bbf505b716
num_q all 300
num_ret all 29931
num_rel all 339
num_rel_ret all 297
map all 0.4657
gm_map all 0.0918
Rprec all 0.3512
bpref all 0.8759
recip_rank all 0.4773
iprec_at_recall_0.00 all 0.4774
iprec_at_recall_0.10 all 0.4774
iprec_at_recall_0.20 all 0.4774
iprec_at_recall_0.30 all 0.4773
iprec_at_recall_0.40 all 0.4762
iprec_at_recall_0.50 all 0.4735
iprec_at_recall_0.60 all 0.4735
iprec_at_recall_0.70 all 0.4704
iprec_at_recall_0.80 all 0.4577
iprec_at_recall_0.90 all 0.4544
iprec_at_recall_1.00 all 0.4544
P_5 all 0.1280
P_10 all 0.0733
P_15 all 0.0547
P_20 all 0.0427
P_30 all 0.0303
P_100 all 0.0099
P_200 all 0.0049
P_500 all 0.0020
P_1000 all 0.0010
