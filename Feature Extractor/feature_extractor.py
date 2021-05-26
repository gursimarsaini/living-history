from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import nltk
import numpy as np


def count_vect(corpus, min_doc_freq, no_of_grams):
    '''this function takes corpus, minimum document frequency
    with respect to a word and the range of number of grams of words
    as arguements and return the count vectorizer learnt from the data'''
    vectorizer = CountVectorizer(min_df=min_doc_freq, ngram_range=no_of_grams)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def tfidf_vect(corpus, min_doc_freq, no_of_grams):
    '''this function takes corpus, minimum document frequency
    with respect to a word and the range of number of grams of words
    as arguements and returns the tfidf vectorizer learnt from the data'''
    vectorizer = TfidfVectorizer(min_df=min_doc_freq, ngram_range=no_of_grams)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def word_2_vec_model(tokenized_corpus, word_size, word_df, window_size):
    '''this function takes four arguements: corpus, word_size for word 
    dimensions, word_df for minimum document frequency for word and a window
    size and then learns a word3vec model for given corpus and returns a word2vec vectorizer'''
    #tokenized_corpus = [nltk.word_tokenize(doc) for doc in corpus]
    model = gensim.models.Word2Vec(tokenized_corpus, size=word_size, min_count=word_df, window=window_size)
    return model


def tfidf_avg_wtd_word_vectors(words, tfidf_vectorizer, tfidf_vocabulary, model, num_features):
    '''this function takes in vectorized document, tfidf features,
    tfidf vocabulary, word2vec model learnt from the data and number
    of features for the dimensionality of document and returns tfidf 
    average weighted word vectors'''
    word_tfidfs = [tfidf_vectorizer[0, tfidf_vocabulary.get(word)] if tfidf_vocabulary.get(word) else 0 for word in words]
    word_tfidf_map = {word:tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}
    feature_vector = np.zeros((num_features,), dtype='float64')
    vocabulary = set(model.index2word)
    weights = 0
    for word in words:
        if word in vocabulary:
            word_vector = model[word]
            weighted_word_vectors = word_tfidf_map[word] * word_vector
            weights = weights + word_tfidf_map[word]
            feature_vector = np.add(feature_vector, weighted_word_vectors)
    if weights:
        feature_vector = np.divide(feature_vector, weights)
    return feature_vector

#this function is genralized version of previous function for complete corpus
def tfidf_avg_wtd_word_vectorizer(corpus, tfidf_vectors, tfidf_vocabulary, num_features, word_size, word_df, window_size):
    '''this function is genralized version of tfidf_avg_wtd_word_vectors
    function for complete corpus '''
    tokenized_corpus = [nltk.word_tokenize(document) for document in corpus]
    model = word_2_vec_model(tokenized_corpus, word_size, word_df, window_size)
    document_tfidf = [(doc, doc_tfidf) for doc, doc_tfidf in zip(tokenized_corpus, tfidf_vectors)]
    features = [tfidf_avg_wtd_word_vectors(vectorized_doc, tfidf, tfidf_vocabulary, model, num_features) for vectorized_doc, tfidf in document_tfidf]
    return features

    
