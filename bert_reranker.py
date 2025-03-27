from sentence_transformers import SentenceTransformer, util
import jsonlines
from tqdm import tqdm

# Load a lightweight sentence transformer model
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")


def load_queries(filepath):
    queries = {}
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            if int(obj["_id"]) % 2 == 1:
                queries[str(obj["_id"])] = obj["text"]
    return queries


def load_corpus(filepath):
    corpus = {}
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            corpus[str(obj["_id"])] = obj["text"]
    return corpus


def load_bm25_results(filepath):
    bm25_results = {}
    with open(filepath, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 6 or parts[0].lower() == "query_id":
                continue

            qid = parts[1]
            doc_id = parts[3]

            try:
                if int(qid) % 2 == 1:
                    bm25_results.setdefault(qid, []).append(doc_id)
            except ValueError:
                continue

    print(f"Loaded BM25 results for {len(bm25_results)} odd-numbered queries.")
    return bm25_results


# Modified rerank with top_k parameter
def rerank(queries, corpus, initial_results, top_k=20):
    all_results = []

    for i, qid in enumerate(tqdm(initial_results)):
        if qid not in queries:
            continue

        query = queries[qid]
        doc_ids = initial_results[qid][:top_k]  # Limit how many docs to rerank

        valid_docs = [
            (doc_id, corpus[doc_id]) for doc_id in doc_ids if doc_id in corpus
        ]
        if not valid_docs:
            print(f"Skipping query {qid} — no valid docs.")
            continue

        doc_ids_filtered, doc_texts = zip(*valid_docs)

        embeddings = model.encode(
            [query] + list(doc_texts), convert_to_tensor=True, batch_size=32
        )
        query_emb = embeddings[0]
        doc_embs = embeddings[1:]

        scores = util.cos_sim(query_emb, doc_embs)[0]
        ranked = sorted(zip(doc_ids_filtered, scores), key=lambda x: x[1], reverse=True)

        for rank, (doc_id, score) in enumerate(ranked[:100]):
            all_results.append(
                f"{qid} Q0 {doc_id} {rank + 1} {score.item():.4f} miniLM_rerank"
            )

        if i % 5 == 0:
            print(f"Processed query {qid} — top doc: {ranked[0][0]}")

    return all_results


def save_results(results, path="Results_miniLM.txt"):
    with open(path, "w") as f:
        for line in results:
            f.write(line + "\n")


# Run it
if __name__ == "__main__":
    queries = load_queries("dataset/queries.jsonl")
    corpus = load_corpus("dataset/corpus.jsonl")
    bm25 = load_bm25_results("bm25_results.txt")

    # Optional: subset for testing
    subset_bm25 = dict(list(bm25.items())[:10])  # only 10 queries

    results = rerank(queries, corpus, subset_bm25, top_k=50)
    save_results(results)

    print("Sample saved to Results_miniLM.txt")
