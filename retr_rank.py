import json
import preprocess as ps
import math
import index as index
import itertools
import uuid
import pandas as pd
from tabulate import tabulate

def load_inverted_index(file):
    """Loads the inverted index from file"""
    f = open(file)
    return f
def load_json(file):
    data = json.load(file)
    file.close()
    return data

def calculate_length_of_document(tf_idf_term_dict):
    """Calculates the length of each document vector """
    """Returns a dictionary indexed by document id"""

    document_length_dict = {}
    for i in tf_idf_term_dict: # each term
        doc_ids = tf_idf_term_dict[i] #get the documents
        for d in doc_ids: #for each document for the term
           if d in document_length_dict:
            document_length_dict[d] = document_length_dict[d] + (doc_ids[d]**2)
           else:
            document_length_dict[d] = (doc_ids[d]**2)
    for d in document_length_dict:
        document_length_dict[d] = math.sqrt(document_length_dict[d])
    
    return document_length_dict
        
def calculate_length_of_query(query_dict):
    """Calculates the length of query from the query dict"""
    
    query_length = 0
    for q in query_dict:
        query_length = query_length + (query_dict[q] **2)
    return math.sqrt(query_length)


def get_document_count(corpus):
    """Given the corpus returns the size of the corpus"""

    doc_count = 0
    ##count the number of document in corpus
    with open(corpus, 'r', encoding='utf-8') as file:
        for line in file:
            doc = json.loads(line.strip())
            doc_id = doc['_id']
            if doc_id:
                doc_count = doc_count+1
    return doc_count

def get_term_idf(term, inverted_index, doc_count):
    """Calculates the idf for the term passed using the inverted index and doc_count"""

    idf = 0
    for t in inverted_index:
        if t == term:
            df_term = len(inverted_index[term])
            idf = math.log(doc_count / df_term)
    return idf

 
def calculate_doc_term_tf_idf(inverted_index, doc_count):
    """Calculates the tf-idf weight for each term in the document posting
       Normalized using the highest term frequency in the inverted index
    """
    
    ## get df for each term
    tf_idf_term_dict = {}
    for term in inverted_index:
        max_tf_freq = 0
        doc_ids = inverted_index[term]
        for doc in doc_ids: ## {doc: val, ....}
            if doc_ids[doc] > max_tf_freq:
                max_tf_freq = doc_ids[doc]
            
        idf = get_term_idf(term, inverted_index, doc_count)
        
        for doc in doc_ids: ## {doc: val, ....}
            term_freq = doc_ids[doc]
            
            if term not in tf_idf_term_dict:
                tf_idf_term_dict[term]  = {doc : ((term_freq/ max_tf_freq) * idf)}
            else:
                tf_idf_term_dict[term][doc]= (term_freq/ max_tf_freq) * idf
    return tf_idf_term_dict
        
def calculate_query_idf(query_vocab_tokens, query_line, inverted_index, vocab, doc_count):
    """Calculates the query idf"""
    """Normalized using the length of unique number of query tokens"""

    query_line_tokens = query_line.split()
    query_dict = {}
    #print(len(set(query_vocab_tokens)))
    #print(len(query_vocab_tokens))
    for q_word in set(query_vocab_tokens):
       if q_word in vocab:
        idf = get_term_idf(q_word,inverted_index, doc_count)
        q_word_tf = query_vocab_tokens.count(q_word)
        query_dict[q_word] = (q_word_tf/ len(set(query_vocab_tokens))) * idf
    return query_dict
    
def calculate_cosine_similarity(query_dict_idf, inverted_index, tf_idf_term_doc_dict):
    """Calculates the  cosine similarity for query"""
    
    document_length_dict = calculate_length_of_document(tf_idf_term_doc_dict) # [dn] = [val]
    len_query = calculate_length_of_query(query_dict_idf)
    doc_q_dict_sim = {}
    ##start here##  
    for q in query_dict_idf:
        docs = inverted_index[q] ## [doc_id: w,...  ]
        #print(q , docs)
        for d in docs:
            #print(tf_idf_term_doc_dict[q][d])
            dot_q_d_t = query_dict_idf[q] * tf_idf_term_doc_dict[q][d]
            if d not in doc_q_dict_sim:
                doc_q_dict_sim[d]= dot_q_d_t
            else:
                doc_q_dict_sim[d] = doc_q_dict_sim[d] + dot_q_d_t ## sum of all the weighted dot product
    
    for d in doc_q_dict_sim:
        len_of_d = document_length_dict[d]
        doc_q_dict_sim[d] = doc_q_dict_sim[d] / (len_of_d * len_query)
    
    doc_q_dict_sim = {k: v for k, v in sorted(doc_q_dict_sim.items(), key=lambda item: item[1], reverse=True)}
    doc_q_dict_sim = dict(itertools.islice(doc_q_dict_sim.items(), 100))      
    
    return doc_q_dict_sim  
    

def rank_documents(queries_file, inverted_index, vocab, tf_idf_term_doc_dict, doc_count ):
    """Ranks the document in descending order of cosine similarity for all queries"""
    
    query_document_sorted_dict = {} 
    with open(queries_file, 'r', encoding='utf-8') as file:
        for line in file:
            # parse JSON line
            query_line = json.loads(line.strip())
            #preprocess the tokens
            query_vocab_tokens = ps.preprocess_document(query_line)
            query_id = query_line.get("_id")

            query_dict_idf = calculate_query_idf(query_vocab_tokens, query_line.get("text"), inverted_index, vocab, doc_count)
            sim_sorted = calculate_cosine_similarity(query_dict_idf, inverted_index, tf_idf_term_doc_dict)
            query_document_sorted_dict[query_id] = sim_sorted
            
    return query_document_sorted_dict


def print_results(rank_dict):
    """Writes the output of the ranking result to a file 
    Note: this procedure is currently taking a long time to complete"""
    
    result_df_columns = ["query_id", "Q0", "doc_id", "rank", "score", "tag"]
    result_df = pd.DataFrame()
    
    run_name = str(uuid.uuid4())
    for q in rank_dict:
        count = 1
        for d in rank_dict[q]:
            data = pd.DataFrame({"query_id": [q], "Q0": ["Q0"], "doc_id": [d], "rank": [count], "score": [rank_dict[q][d]], "tag": [run_name]})
            result_df = pd.concat([result_df, data], ignore_index=True)
            count = count +1
    
    with open('Results.txt', 'w') as r_file:
        r_file.write(tabulate(result_df, headers = 'keys', tablefmt = 'plain'))


if __name__ == "__main__":
    document_count = get_document_count("dataset/corpus.jsonl")
    
    print("Loading vocabulary")
    vocab = index.load_vocabulary("vocabulary.txt")

    print("Loading inverted index")
    file = load_inverted_index('inverted_index.json')
    inverted_index = load_json(file)

    print("Calculating tf_idf for each term in corpus")
    tf_idf_term_doc_dict = calculate_doc_term_tf_idf(inverted_index, document_count)
    
    queries_file = "dataset/queries.jsonl"

    print("Starting to rank documents, this may take a while...")
    rank_dict = rank_documents(queries_file, inverted_index,  vocab, tf_idf_term_doc_dict, document_count)
    print("Ranking calculation done")

    print("Printing results to file, this may take a while...")
    print_results(rank_dict)

    print("Program done. Check the Results.txt for the output")
    