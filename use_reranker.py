import tensorflow_hub as hub
import tensorflow as tf
import jsonlines
from tqdm import tqdm
import numpy as np
from sentence_transformers import util

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5", "https://tfhub.dev/google/universal-sentence-encoder-lite/2"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)

def load_queries(filepath):
    queries = {}
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            queries[str(obj["_id"])] = obj["text"]
    return queries


def load_corpus(filepath):
    corpus = {}
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            corpus[str(obj["_id"])] = obj["text"] + " " + obj["title"]
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

            bm25_results.setdefault(qid, []).append(doc_id)

    print(f"Loaded BM25 results for {len(bm25_results)} queries.")
    return bm25_results


def get_test_qids(qrel_path):
    qids = set()
    with open(qrel_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                qids.add(parts[0])
    print(f"Found {len(qids)} query IDs in test.qrel.")
    return qids


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


        scores = []
        
        query_vec = model([query])
        for doc_text in doc_texts:
           sim = util.cos_sim(tf.constant(query_vec[0]).numpy(), tf.constant(model([doc_text])[0]).numpy())[0][0]
           scores.append(sim)

        ranked = sorted(zip(doc_ids_filtered, scores), key=lambda x: x[1], reverse=True)

        for rank, (doc_id, score) in enumerate(ranked[:100]):
            all_results.append(
                f"{qid} Q0 {doc_id} {rank + 1} {score.item():.4f} useLM_rerank"
            )

        if i % 5 == 0:
            print(f"Processed query {qid} — top doc: {ranked[0][0]}")

    return all_results





def save_results(results, path="Results_useLM.txt"):
    with open(path, "w") as f:
        for line in results:
            f.write(line + "\n")




# Run it
if __name__ == "__main__":
    queries = load_queries("dataset/queries.jsonl")
    corpus = load_corpus("dataset/corpus.jsonl")
    bm25 = load_bm25_results("bm25_results.txt")
    test_qids = get_test_qids("dataset/test.qrel")


    # Filter BM25 results to test.qrel queries only
    filtered_bm25 = {qid: docs for qid, docs in bm25.items() if qid in test_qids}
    print(f"Filtered to {len(filtered_bm25)} BM25 queries matching test.qrel.")

    results = rerank(queries, corpus, filtered_bm25, top_k=100  )
    save_results(results)

    print("Results saved to Results_useLM.txt")