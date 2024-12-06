from sentence_transformers import SentenceTransformer
from typing import Dict
import nltk
import re 
from collections import Counter
from nltk.corpus import stopwords



def bag_of_words(titles):
    nltk.download('stopwords')
    nltk.download('punkt_tab')

    def tokenize(text) -> Dict[str, int]:
        dataset = nltk.sent_tokenize(text) 
        for i in range(len(dataset)): 
            dataset[i] = dataset[i].lower() 
            dataset[i] = re.sub(r'\W', ' ', dataset[i]) 
            dataset[i] = re.sub(r'\s+', ' ', dataset[i]) 
        word2count = {} 
        for data in dataset: 
            words = nltk.word_tokenize(data) 
            for word in words: 
                if word not in word2count.keys(): 
                    word2count[word] = 1
                else: 
                    word2count[word] += 1
        return word2count

    # Get Counts across entire dataset
    word2count = Counter({})
    for title in titles:
        word2count.update(tokenize(title))
    
    for word in stopwords.words('english'):
        del word2count[word]

    return word2count



def sbert_embedding(text, device="mps"):
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    t = model.encode(text)
    return t


import nltk
from nltk.corpus import stopwords
from collections import Counter
import re

def bag_of_words_with_laplace(text, alpha=1):
    nltk.download('stopwords')
    nltk.download('punkt')

    def tokenize(text) -> Dict[str, int]:
        dataset = nltk.sent_tokenize(text)
        for i in range(len(dataset)):
            dataset[i] = dataset[i].lower()
            dataset[i] = re.sub(r'\W', ' ', dataset[i])
            dataset[i] = re.sub(r'\s+', ' ', dataset[i])
        word2count = {}
        for data in dataset:
            words = nltk.word_tokenize(data)
            for word in words:
                if word not in word2count:
                    word2count[word] = 1
                else:
                    word2count[word] += 1
        return word2count

    # Get the full vocabulary
    vocabulary = set()
    for title in text:
        vocabulary.update(tokenize(title).keys())
    for description in text:
        vocabulary.update(tokenize(description).keys())
    
    # Remove stop words from the vocabulary
    stop_words = set(stopwords.words('english'))
    vocabulary = vocabulary - stop_words
    vocabulary = list(vocabulary)

    # Initialize word counts with Laplace smoothing
    vocab_size = len(vocabulary)
    vocabulary_dict = {word: i for i, word in enumerate(vocabulary)}

    def compute_bow_with_smoothing(texts):
        bow_matrix = []
        for text in texts:
            word_counts = Counter(tokenize(text))
            bow_vector = [alpha] * vocab_size  # Initialize with Laplace smoothing
            for word, count in word_counts.items():
                if word in vocabulary_dict:
                    word_idx = vocabulary_dict[word]
                    bow_vector[word_idx] += count
            bow_matrix.append(bow_vector)
        return bow_matrix

    # Compute Bag of Words for titles and descriptions
    bow = compute_bow_with_smoothing(text)

    return bow, vocabulary
