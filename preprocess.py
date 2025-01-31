import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from nltk.tokenize import word_tokenize
import nltk
from bs4 import BeautifulSoup

# download NLTK resources (stopwords and punkt tokenizer)
nltk.download('stopwords')
nltk.download('punkt')

# initialize stemmer
stemmer = PorterStemmer()

# removing punctuation and numbers
def clean_text(text):
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text.lower()

# removing HTML/XML mark up
def remove_markup(text):
    return BeautifulSoup(text, "html.parser").get_text()

# splitting text into individual words
def tokenize(text):
    return word_tokenize(text.lower())

# filtering out common meaningless words
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words and len(token) > 2]   # discard very short words

#reducing words to root forms
def stem(tokens):
    return [stemmer.stem(token) for token in tokens]

def preprocess_document(doc):
    text = doc.get('text', '')
    
    # clean text (remove markup, punctuation, numbers)
    text = remove_markup(text)
    text = clean_text(text)
    
    # tokenize the text
    tokens = tokenize(text)
    
    # remove stopwords
    tokens = remove_stopwords(tokens)
    
    # apply stemming 
    tokens = stem(tokens)
    
    return tokens

def preprocess_corpus(corpus_file):
    preprocessed_docs = []
    
    with open(corpus_file, 'r', encoding='utf-8') as file:
        for line in file:
            # parse JSON line
            doc = json.loads(line.strip())
            
            # preprocess the document
            tokens = preprocess_document(doc)
            
            # add to the list of preprocessed documents
            preprocessed_docs.append(tokens)
    
    return preprocessed_docs

# for testing/debugging 
def save_vocabulary(preprocessed_docs, output_file):
    vocabulary = set(token for doc in preprocessed_docs for token in doc)  # flatten list of tokens
    
    # save vocabulary to a file (sorted)
    with open(output_file, 'w', encoding='utf-8') as file:
        for token in sorted(vocabulary):
            file.write(token + '\n')

    print(f"Vocabulary saved to {output_file} with {len(vocabulary)} unique tokens.")

if __name__ == "__main__":
    
    corpus_file = "dataset/corpus.jsonl"  # path to the corpus file
    
    preprocessed_docs = preprocess_corpus(corpus_file) # preprocess the corpus
    
    save_vocabulary(preprocessed_docs, "vocabulary.txt")  # for testing
