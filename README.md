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

## Detailed Descriptions

### Preprocessing (`preprocess.py`)
This script makes sure that the text from our documents is clean and ready for the computer to understand easily. It involves removing unnecessary clutter, splitting text into words, removing common but unhelpful words, converting words to their base forms, and standardizing text.

### Indexing (`index.py`)
This script organizes the cleaned text so that it can quickly find documents when you search for specific words. It creates a list of all unique words and builds an inverted index that details where each word appears in the documents and how often.

### Retrieval & Ranking (`retr_rank.py`)
This script looks at your search words, finds documents that contain those words, and arranges them by how relevant they are to your search. It processes search queries, finds matching documents, and ranks documents by relevance using a scoring system based on term frequency and document similarity.


## Viewing the Output

The created inverted index is saved in `inverted_index.json`. Open this file with any JSON-supporting text editor to examine the structured index.

The ranking of the documents is stored in `Results.txt`. Open this file in a text editor to view the ranking of the documents for the queries.

### **Algorithms, Data Structures, and Optimizations Used**

---

### **1. Preprocessing (`preprocess.py`)**
This step processes raw text to make it ready for indexing and retrieval.

- Algorithms:
  - HTML/XML markup removal using BeautifulSoup.
  - Tokenization using NLTK’s `word_tokenize`.
  - Stopword removal with NLTK stopword list.
  - Stemming using Porter Stemmer to reduce words to their root form.
  - Text normalization by converting to lowercase and removing punctuation and numbers.

- Data Structures:
  - Lists: Used to store tokenized words for each document.
  - Sets: Used for the stopword list (fast lookups) and for storing unique vocabulary words.

- Optimizations:
  - Uses compiled regular expressions (`re.sub`) for efficient text cleaning.
  - NLTK stopwords stored in a set for faster membership checks.

---

### **2. Indexing (`index.py`)**
This step builds an inverted index for fast retrieval.

- Algorithms:
  - Inverted Index Construction: Each term maps to a dictionary of `{doc_id: term_frequency}`.
  - Tokenization and Cleaning: Uses `preprocess_document()` from `preprocess.py`.

- Data Structures:
  - Set: Used for storing vocabulary for fast lookup.
  - Dictionary (`dict`):
    - `{term: {doc_id: frequency}}` structure for inverted index.
    - Nested dictionaries for term-document frequency tracking.

- Optimizations:
  - Early filtering: Only indexes words found in the vocabulary to reduce index size.
  - Efficient JSON processing using `json.loads(line.strip())` for handling large datasets.
  - Index saved in JSON format for easy loading in the retrieval step.

---

### **3. Retrieval & Ranking (`retr_rank.py`)**
This step retrieves and ranks relevant documents based on TF-IDF and Cosine Similarity.

- Algorithms:
  - TF-IDF Calculation:
    - Uses `term frequency (tf)` and inverse document frequency `(idf)` to weight terms.
    - Normalization: Term frequency is scaled by the highest term frequency in each document.
  - Cosine Similarity:
    - Computes dot product between query and document vectors.
    - Uses precomputed document vector lengths for efficiency.
  - Sorting & Ranking:
    - Uses sorted dictionary (`lambda sort`) to rank documents.

- Data Structures:
  - Dictionary (`dict`):
    - `{term: {doc_id: tf-idf score}}` for TF-IDF term-document representation.
    - `{query_id: {doc_id: similarity_score}}` for ranking.
  - Lists: Used to store preprocessed queries.
  - Pandas DataFrame: Used for formatting ranking results.

- Optimizations:
  - Precalculates document lengths to avoid repeated calculations.
  - Keeps top 100 documents using `itertools.islice()` for efficiency.
  - Batch ranking and writing to file to reduce I/O overhead.


## Vocabulary
Our vocabulary was had a length of 30871. <br>

Here is a sample of 100 tokens from the vocabulary: <br>

aa
aaa
aaaatpas
aaafamili
aab
aabenhu
aacr
aacrthi
aad
aadinduc
aadtreat
aag
aah
aai
aakampk
aalpha
aam
aanatsnat
aarhu
aaronquinlangmailcom
aasv
aatf
aauaaa
aav
ab
abad
abandon
abas
abb
abber
abbott
abbrevi
abc
abca
abcamedi
abcb
abcc
abcg
abciximab
abciximabrel
abctarget
abd
abda
abdb
abdomen
abdomin
abdominala
abduct
aberr
aberrantincomplet
aberrantli
abeta
abf
abfreb
abfrebsit
abi
abibas
abil
abiot
abirateron
abl
ablat
abm
abmd
abnorm
abocompat
abolish
abort
abound
abovefacil
abovement
abp
abpa
abpi
abrb
abroad
abrog
abrupt
abruptli
abscess
abscis
absciss
absenc
absent
absolut
absoluteconcentr
absorb
absorbancecytophotometr
absorpt
absorptiometri
abstain
abstent
abstin
abstract
abstracta
abstractmicrorna
abt
abuja
abulia
abund
<br>

### First 10 Answers for the First Two Queries:

**Query ID: 0**
1. **13231899**
2. **1836154**
3. **42373087**
4. **25301182**
5. **10342807**
6. **24660385**
7. **14827874**
8. **42731834**
9. **994800**
10. **4435369**

**Query ID: 2**
1. **13734012**
2. **14610165**
3. **26059876**
4. **11992632**
5. **9038803**
6. **13770184**
7. **32922179**
8. **21855837**
9. **8509018**
10. **1203035**
<br>
Analysis: The results for Query ID: 0 and Query ID: 2 are completely different from one another, meaning that each query retrieves a unique set of top-ranked documents.
This suggests that the vocabulary filtering and token weighting (TF-IDF) are effective in distinguishing different queries.

#### Results for Query ID: 0
The documents that came up for this query matched closely with the topics and keywords we asked about. Our system uses a combination of TF-IDF and cosine similarity to find documents that really fit what the search query is about. This means it’s not just finding random documents; it’s finding the ones that are most relevant.

#### Results for Query ID: 2
Similar to the first query, the documents retrieved here were highly relevant to the terms searched. This shows our system’s ability to not only find documents that contain the search terms but to rank them in a way that the most relevant ones come first.

#### Key Observations and System Effectiveness

Effectiveness: Seeing completely different sets of documents for each query suggests our system is really focusing on the specifics of each search request. It adjusts its responses based on what you're looking for, which is a great sign of a responsive search system.

Patterns: In both queries, the documents with a higher presence of the searched terms ranked higher. This is expected and shows our system is doing its job well.

Room for Improvement: Even though the system is performing well, we noticed some variation in how well different documents matched the query. This might mean we need to tweak how we calculate the importance of different terms, possibly by adjusting our formulas a bit.

## Results of trec_eval

Steps to run:

1. Open the evaluate.ipynb file in your Jupyter Notebook environment.
2. Upload the following files to the notebook: Results.txt, Results_title_text.txt, Results_title.txt test.tsv
3. Run the cells.

Below are the key results for corpus text only: <br>
runid all 5d7d54f3-bbcd-4a7b-ae8d-78bbf505b716 <br>
num_q all 300 <br>
num_ret all 29931 <br>
num_rel all 339 <br>
num_rel_ret all 297 <br>
map all 0.4657 <br>
gm_map all 0.0918 <br>
Rprec all 0.3512 <br>
bpref all 0.8759 <br>
recip_rank all 0.4773 <br>
iprec_at_recall_0.00 all 0.4774 <br>
iprec_at_recall_0.10 all 0.4774 <br>
iprec_at_recall_0.20 all 0.4774 <br>
iprec_at_recall_0.30 all 0.4773 <br>
iprec_at_recall_0.40 all 0.4762 <br>
iprec_at_recall_0.50 all 0.4735 <br>
iprec_at_recall_0.60 all 0.4735 <br>
iprec_at_recall_0.70 all 0.4704 <br>
iprec_at_recall_0.80 all 0.4577 <br>
iprec_at_recall_0.90 all 0.4544 <br>
iprec_at_recall_1.00 all 0.4544 <br>
P_5 all 0.1280 <br>
P_10 all 0.0733 <br>
P_15 all 0.0547 <br>
P_20 all 0.0427 <br>
P_30 all 0.0303 <br>
P_100 all 0.0099 <br>
P_200 all 0.0049 <br>
P_500 all 0.0020 <br>
P_1000 all 0.0010 <br>

The Mean Average Precision (MAP) score for our retrieval system with corpus text, using trec_eval, is 0.4657. This shows a moderate effectiveness in retrieving relevant documents, as MAP measures the quality of ranked results by averaging precision across recall levels.

Below are the key results for corpus title only: <br>

runid                 	all	aaa1cfe0-4bd9-4fc2-abf6-98c35ab6854e <br>
num_q                 	all	300 <br>
num_ret               	all	28583 <br>
num_rel               	all	339 <br>
num_rel_ret           	all	235 <br>
map                   	all	0.3520 <br>
gm_map                	all	0.0121 <br>
Rprec                 	all	0.2646 <br>
bpref                 	all	0.6807 <br>
recip_rank            	all	0.3680 <br>
iprec_at_recall_0.00  	all	0.3683 <br>
iprec_at_recall_0.10  	all	0.3683 <br>
iprec_at_recall_0.20  	all	0.3683 <br>
iprec_at_recall_0.30  	all	0.3680 <br>
iprec_at_recall_0.40  	all	0.3628 <br>
iprec_at_recall_0.50  	all	0.3558 <br>
iprec_at_recall_0.60  	all	0.3558 <br>
iprec_at_recall_0.70  	all	0.3551 <br>
iprec_at_recall_0.80  	all	0.3413 <br>
iprec_at_recall_0.90  	all	0.3406 <br>
iprec_at_recall_1.00  	all	0.3406 <br>
P_5                   	all	0.0973 <br>
P_10                  	all	0.0547 <br>
P_15                  	all	0.0396 <br>
P_20                  	all	0.0307 <br>
P_30                  	all	0.0218 <br>
P_100                 	all	0.0078 <br>
P_200                 	all	0.0039 <br>
P_500                 	all	0.0016 <br>
P_1000                	all	0.0008 <br>

The Mean Average Precision (MAP) score for our retrieval system with corpus title, using trec_eval, is 0.3520.

Below are the key results for corpus title and text: <br>

runid                 	all	a06e8ec5-c2ec-443b-8cc5-142febf16054 <br>
num_q                 	all	300 <br>
num_ret               	all	29933 <br>
num_rel               	all	339 <br>
num_rel_ret           	all	304 <br>
map                   	all	0.4842 <br>
gm_map                	all	0.1139 <br>
Rprec                 	all	0.3679 <br>
bpref                 	all	0.8949 <br>
recip_rank            	all	0.4957 <br>
iprec_at_recall_0.00  	all	0.4959 <br>
iprec_at_recall_0.10  	all	0.4959 <br>
iprec_at_recall_0.20  	all	0.4959 <br>
iprec_at_recall_0.30  	all	0.4958 <br>
iprec_at_recall_0.40  	all	0.4936 <br>
iprec_at_recall_0.50  	all	0.4915 <br>
iprec_at_recall_0.60  	all	0.4915 <br>
iprec_at_recall_0.70  	all	0.4894 <br>
iprec_at_recall_0.80  	all	0.4771 <br>
iprec_at_recall_0.90  	all	0.4733 <br>
iprec_at_recall_1.00  	all	0.4733 <br>
P_5                   	all	0.1347 <br>
P_10                  	all	0.0773 <br>
P_15                  	all	0.0556 <br>
P_20                  	all	0.0430 <br>
P_30                  	all	0.0306 <br>
P_100                 	all	0.0101 <br>
P_200                 	all	0.0051 <br>
P_500                 	all	0.0020 <br>
P_1000                	all	0.0010 <br>

The Mean Average Precision (MAP) score for our retrieval system with corpus title and text, using trec_eval, is 0.4842.

The MAP score for text and title is higher than only text or only title. The MAP score is better as combining both title and text increases cosine similarity score as the query terms have higher chances to be matched with vocabulary terms.




## Contributions & Tasks Divided

Shabrina Sharmin (300230297): retr_rank.py (Retrieval and Ranking), report.

Shannon Noah (300163898): Preprocessing.py (Preprocessing), and computing evaluation measures using trec_eval script, report.

Tina Trinh (300175427): Index.py (Indexing step), report.
