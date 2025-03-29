from sentence_transformers import SentenceTransformer, util
import jsonlines
from tqdm import tqdm

# Load a lightweight sentence transformer model fine-tuned for QA retrieval tasks.
# This model generates dense vector embeddings for semantic similarity.
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")


# Load queries from the JSONL file and store them in a dictionary using their _id as key.
def load_queries(filepath):
    queries = {}
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            queries[str(obj["_id"])] = obj["text"]
    return queries


# Load the document corpus from JSONL, with doc _id as key and text content as value.
def load_corpus(filepath):
    corpus = {}
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            corpus[str(obj["_id"])] = obj["text"]
    return corpus


# Load BM25 ranking results (from Assignment 1) in TREC format.
# Each line includes a query ID and a corresponding document ID.
def load_bm25_results(filepath):
    bm25_results = {}
    with open(filepath, "r") as file:
        for line in file:
            parts = line.strip().split()
            # Skip header or malformed lines
            if len(parts) < 6 or parts[0].lower() == "query_id":
                continue

            qid = parts[1]  # Query ID is in the 2nd column
            doc_id = parts[3]  # Document ID is in the 4th column

            bm25_results.setdefault(qid, []).append(doc_id)

    print(f"Loaded BM25 results for {len(bm25_results)} queries.")
    return bm25_results


# Load query IDs from the test.qrel file — these are the queries used for final evaluation.
def get_test_qids(qrel_path):
    qids = set()
    with open(qrel_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                qids.add(parts[0])
    print(f"Found {len(qids)} query IDs in test.qrel.")
    return qids


# Rerank BM25 results using MiniLM model based on cosine similarity.
def rerank(queries, corpus, initial_results, top_k=20):
    all_results = []

    for i, qid in enumerate(tqdm(initial_results)):
        if qid not in queries:
            continue

        query = queries[qid]
        doc_ids = initial_results[qid][
            :top_k
        ]  # Only rerank the top-k documents per query

        # Ensure the documents exist in the corpus
        valid_docs = [
            (doc_id, corpus[doc_id]) for doc_id in doc_ids if doc_id in corpus
        ]
        if not valid_docs:
            print(f"Skipping query {qid} — no valid docs.")
            continue

        doc_ids_filtered, doc_texts = zip(*valid_docs)

        # Encode query and documents as dense embeddings
        embeddings = model.encode(
            [query] + list(doc_texts), convert_to_tensor=True, batch_size=32
        )
        query_emb = embeddings[0]
        doc_embs = embeddings[1:]

        # Compute cosine similarity between query and each document
        scores = util.cos_sim(query_emb, doc_embs)[0]

        # Sort documents based on descending similarity score
        ranked = sorted(zip(doc_ids_filtered, scores), key=lambda x: x[1], reverse=True)

        # Format result lines in TREC format for top 100 ranked docs
        for rank, (doc_id, score) in enumerate(ranked[:100]):
            all_results.append(
                f"{qid} Q0 {doc_id} {rank + 1} {score.item():.4f} miniLM_rerank"
            )

        # Print progress every 5 queries
        if i % 5 == 0:
            print(f"Processed query {qid} — top doc: {ranked[0][0]}")

    return all_results


# Save the final re-ranked results to a file in TREC format
def save_results(results, path="Results_miniLM.txt"):
    with open(path, "w") as f:
        for line in results:
            f.write(line + "\n")


# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    # Load queries, corpus, and BM25 results
    queries = load_queries("dataset/queries.jsonl")
    corpus = load_corpus("dataset/corpus.jsonl")
    bm25 = load_bm25_results("bm25_results.txt")

    # Filter BM25 results to only include queries present in test.qrel
    test_qids = get_test_qids("dataset/test.qrel")
    filtered_bm25 = {qid: docs for qid, docs in bm25.items() if qid in test_qids}
    print(f"Filtered to {len(filtered_bm25)} BM25 queries matching test.qrel.")

    # Run reranking with MiniLM on top 50 BM25 docs per query
    results = rerank(queries, corpus, filtered_bm25, top_k=50)

    # Save final results to file
    save_results(results)
    print("Results saved to Results_miniLM.txt")
