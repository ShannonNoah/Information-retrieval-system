import json
import preprocess as ps
import math
import index as index
import itertools
import uuid
import pandas as pd
from tabulate import tabulate


# Loads the inverted index from a JSON file
def load_inverted_index(file):
    f = open(file)
    return f


# Parses a JSON file
def load_json(file):
    data = json.load(file)
    file.close()
    return data


# Computes the length (L2 norm) of each document's TF-IDF vector
def calculate_length_of_document(tf_idf_term_dict):
    document_length_dict = {}
    for term in tf_idf_term_dict:
        for doc_id in tf_idf_term_dict[term]:
            weight = tf_idf_term_dict[term][doc_id]
            document_length_dict[doc_id] = (
                document_length_dict.get(doc_id, 0) + weight**2
            )
    for doc_id in document_length_dict:
        document_length_dict[doc_id] = math.sqrt(document_length_dict[doc_id])
    return document_length_dict


# Computes the length (L2 norm) of the query TF-IDF vector
def calculate_length_of_query(query_dict):
    return math.sqrt(sum(weight**2 for weight in query_dict.values()))


# Counts the total number of documents in the corpus
def get_document_count(corpus):
    doc_count = 0
    with open(corpus, "r", encoding="utf-8") as file:
        for line in file:
            doc = json.loads(line.strip())
            if doc.get("_id"):
                doc_count += 1
    return doc_count


# Calculates the IDF value for a term
def get_term_idf(term, inverted_index, doc_count):
    if term in inverted_index:
        df_term = len(inverted_index[term])
        return math.log(doc_count / df_term)
    return 0


# Computes the TF-IDF weight for each term in each document
def calculate_doc_term_tf_idf(inverted_index, doc_count):
    tf_idf_term_dict = {}
    for term, doc_freqs in inverted_index.items():
        max_tf = max(doc_freqs.values())
        idf = get_term_idf(term, inverted_index, doc_count)
        for doc_id, freq in doc_freqs.items():
            tf_idf = (freq / max_tf) * idf
            tf_idf_term_dict.setdefault(term, {})[doc_id] = tf_idf
    return tf_idf_term_dict


# Computes TF-IDF weights for a query using IDF from the inverted index
def calculate_query_idf(query_tokens, query_text, inverted_index, vocab, doc_count):
    query_dict = {}
    for word in set(query_tokens):
        if word in vocab:
            idf = get_term_idf(word, inverted_index, doc_count)
            tf = query_tokens.count(word)
            query_dict[word] = (tf / len(set(query_tokens))) * idf
    return query_dict


# Calculates cosine similarity between the query and all documents
def calculate_cosine_similarity(query_dict_idf, inverted_index, tf_idf_term_doc_dict):
    document_length_dict = calculate_length_of_document(tf_idf_term_doc_dict)
    query_length = calculate_length_of_query(query_dict_idf)
    doc_q_sim = {}

    for term in query_dict_idf:
        if term in inverted_index:
            for doc_id in inverted_index[term]:
                dot_product = query_dict_idf[term] * tf_idf_term_doc_dict[term][doc_id]
                doc_q_sim[doc_id] = doc_q_sim.get(doc_id, 0) + dot_product

    for doc_id in doc_q_sim:
        doc_q_sim[doc_id] /= document_length_dict[doc_id] * query_length

    # Sort and return top 100 documents by similarity
    sorted_sim = dict(sorted(doc_q_sim.items(), key=lambda x: x[1], reverse=True))
    return dict(itertools.islice(sorted_sim.items(), 100))


# Processes each query, computes similarities, and stores top results
def rank_documents(
    queries_file, inverted_index, vocab, tf_idf_term_doc_dict, doc_count
):
    query_document_sorted_dict = {}
    with open(queries_file, "r", encoding="utf-8") as file:
        for line in file:
            query_obj = json.loads(line.strip())
            query_tokens = ps.preprocess_document(query_obj)
            query_id = query_obj["_id"]
            query_dict_idf = calculate_query_idf(
                query_tokens, query_obj["text"], inverted_index, vocab, doc_count
            )
            sim_results = calculate_cosine_similarity(
                query_dict_idf, inverted_index, tf_idf_term_doc_dict
            )
            query_document_sorted_dict[query_id] = sim_results
    return query_document_sorted_dict


# Saves ranked documents in TREC format to 'Results.txt'
def print_results(rank_dict):
    result_df_columns = ["query_id", "Q0", "doc_id", "rank", "score", "tag"]
    result_df = pd.DataFrame()
    run_name = str(uuid.uuid4())

    for qid in rank_dict:
        for rank, (doc_id, score) in enumerate(rank_dict[qid].items(), start=1):
            result_df = pd.concat(
                [
                    result_df,
                    pd.DataFrame(
                        {
                            "query_id": [qid],
                            "Q0": ["Q0"],
                            "doc_id": [doc_id],
                            "rank": [rank],
                            "score": [score],
                            "tag": [run_name],
                        }
                    ),
                ],
                ignore_index=True,
            )

    with open("Results.txt", "w") as f:
        f.write(tabulate(result_df, headers="keys", tablefmt="plain"))


# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    document_count = get_document_count("dataset/corpus.jsonl")

    print("Loading vocabulary")
    vocab = index.load_vocabulary("vocabulary.txt")

    print("Loading inverted index")
    file = load_inverted_index("inverted_index.json")
    inverted_index = load_json(file)

    print("Calculating tf_idf for each term in corpus")
    tf_idf_term_doc_dict = calculate_doc_term_tf_idf(inverted_index, document_count)

    print("Starting to rank documents, this may take a while...")
    queries_file = "dataset/queries.jsonl"
    rank_dict = rank_documents(
        queries_file, inverted_index, vocab, tf_idf_term_doc_dict, document_count
    )
    print("Ranking calculation done")

    print("Printing results to file, this may take a while...")
    print_results(rank_dict)

    print("Program done. Check the Results.txt for the output")
