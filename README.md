# CSI4107 - Assignment 2: Neural Information Retrieval System (GROUP 45)

This assignment builds on our classical IR system from Assignment 1 by incorporating neural re-ranking techniques using recent transformer-based and vector-based models. We applied the BM25 ranking system from our inverted index as a base retriever, then re-ranked the top results using advanced embedding-based similarity methods.

Our goal was to investigate whether semantic similarity-based methods using sentence encoders could outperform lexical-based models like BM25 in retrieving the most relevant documents for scientific claims from the Scifact dataset.

---

## **Setup Instructions**

### 1. Install Required Libraries

Ensure Python 3.8 or higher is installed on your system. Install the required Python libraries by running:

```bash
pip install nltk jsonlines pandas sentence-transformers tensorflow tensorflow-hub tqdm gensim
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

Assignment 1:

- **`preprocess.py`**: Script to clean and prepare text data for indexing.
- **`index.py`**: Constructs an inverted index from the preprocessed data.
- **`vocabulary.txt`**: Stores all unique tokens extracted from the text.
- **`dataset`**: Contains the `corpus.jsonl`, `queries.jsonl`, `train.csv`, and `test.csv`.
- **`inverted_index.json`**: The final outputted inverted index.
- **`retr_rank.py`** : Constructs the document ranking based on cosine similarity of the query.
- **`Results_title_text`** : Baseline results from Assignment 1 using title + text.

Assignment 2:

- **`Results.txt`** : Final submission file with results from the best model (MiniLM).
- **`Results_miniLM.txt, Results_useLM.txt, Results_doc2vec.txt`** : Output from respective reranking methods.
- **`Evaluate_Assignment_2.ipynb`** : Runs trec_eval and compares MAP & P@10 across methods.
- **`bert_reranker.py`** : Re-ranks BM25 results using MiniLM (SentenceTransformers).
- **`use_reranker.py`** : Re-ranks BM25 results using Universal Sentence Encoder.
- **`doc2vec_reranker.py`** : Re-ranks BM25 results using a trained Doc2Vec model.

## Usage

### Step 1: Generate Re-Ranked Results

Navigate to the project directory and run:

```bash
python bert_reranker.py

python use_reranker.py

python doc2vec_reranker.py

```

All scripts save their output in a .txt file formatted for trec_eval.

### Step 2: Evaluate with TREC_EVAL

Steps to run:

1. Open the Evaluate_Assignment_2.ipynb file in your Jupyter Notebook environment.
2. Upload the following files to the notebook: test.qrel, Results_title_text.txt, Results_miniLM.txt, Results_useLM.txt, Results_doc2vec.txt`
3. Run the cells.

## ✅ Summary of Enhancements ✅ 

- Used top 50–100 BM25 results from Assignment 1 as the candidate set for re-ranking.
- Applied three re-ranking techniques:
    - MiniLM (SentenceTransformers)
    - Universal Sentence Encoder (USE)
    - Doc2Vec (Gensim)
- Compared results against the original BM25 baseline using trec_eval.
- Highlighted performance based on MAP and P@10.

## Algorithms, Data Structures, and Optimizations

### MiniLM (SentenceTransformers)

- **Algorithm**: For each test query, top-k documents are retrieved from BM25 results and re-ranked using cosine similarity of dense embeddings from MiniLM (via HuggingFace's sentence-transformers).
- **Data Structures**:
  - `dict` for query/document mapping.
  - Lists and tuples to store results and sort scores.
- **Optimizations**:
  - Used batch encoding for performance.
  - Limited reranking to top 50 candidates for speed and relevance.
  - Filtered BM25 results to include only queries from `test.qrel` to ensure alignment with evaluation set and improve MAP scores.

### Universal Sentence Encoder (USE)

- **Algorithm**: Encodes query and candidate documents using TensorFlow Hub’s USE model. Cosine similarity is calculated to rerank candidates.
- **Data Structures**:
  - Python dictionaries and lists.
- **Optimizations**:
  - Uses top 100 BM25 documents per query.
  - Preloaded TensorFlow model reused across all queries.
  - Filtered BM25 results to include only queries from `test.qrel` to ensure alignment with evaluation set and improve MAP scores.

### Doc2Vec

- **Algorithm**: Trains a Doc2Vec model using Gensim on full corpus. Infers vector for each query and reranks documents by cosine similarity with doc vectors.
- **Data Structures**:
  - Gensim `TaggedDocument` format for training.
  - Lists and dictionaries to map and sort data.
- **Optimizations**:
  - Trained with multi-threading (`workers=4`) and 50 epochs for vector quality.
  - Filtered BM25 results to include only queries from `test.qrel` to ensure alignment with evaluation set and improve MAP scores.

---

## Results Discussion

| Method                  | MAP    | P@10   |
| ----------------------- | ------ | ------ |
| MiniLM Rerank           | 0.5077 | 0.0823 |
| Doc2Vec Rerank          | 0.4651 | 0.0733 |
| USE Rerank              | 0.3338 | 0.0573 |
| Baseline (Title + Text) | 0.4842 | 0.0773 |

- **MiniLM** was the most effective reranker overall. It achieved the highest MAP and P@10 scores, indicating that it consistently returned more relevant documents at the top of the list. This model benefits from being trained on a wide variety of question-answer pairs, making it well-suited for tasks like this where queries vary widely.

- **Doc2Vec** performed reasonably well but struggled to understand deeper semantic relationships in text, which is where transformer-based models like MiniLM have an advantage.

- **Universal Sentence Encoder (USE)** had the lowest performance. This is likely because USE is trained to provide general-purpose sentence embeddings rather than task-specific ones. As a result, it wasn’t as effective at ranking documents for complex scientific queries.

- **The Baseline (BM25 with title + text)** actually held up well, beating both USE and Doc2Vec. This shows that even traditional keyword-based models can perform very competitively when provided with high-quality input like combined title and abstract text.

All models were evaluated using `trec_eval` against the official `test.qrel` file to ensure consistency and fair comparison across metrics.

---

### First 10 Answers

Top 10 documents for Query ID 1:

1. 10786948
2. 35008773
3. 16287725
4. 7581911
5. 32001951
6. 42421723
7. 10342807
8. 825728
9. 4430962
10. 680949

Top 10 documents for Query ID 3:

1. 4414547
2. 12271486
3. 4632921
4. 23389795
5. 4378885
6. 19058822
7. 10145528
8. 14717500
9. 3823862
10. 32181055

## Contributions & Tasks Divided

Shabrina Sharmin (300230297): Implemented Universal Sentence Encoder & Doc2Vec model reranker, report.

Shannon Noah (300163898): Added query filtering logic to improve MAP scores, integrated trec_eval, found top 10 answers, report.

Tina Trinh (300175427): Implemented MiniLM reranker, report.
