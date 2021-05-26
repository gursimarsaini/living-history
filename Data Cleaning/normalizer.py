import nltk
import re
import string
from nltk.stem import WordNetLemmatizer


stopword_list = nltk.corpus.stopwords.words('english')
wnl = WordNetLemmatizer()


def text_lower_casing(text):
    '''returns the lowercase text'''
    return text.lower()


def text_tokenizer(text):
    '''return the tokens(words) of a text'''
    tokens = nltk.word_tokenize(text)
    return tokens


def remove_stopwords(text):
    '''removes the stopwords from the text'''
    tokenized_text = text_tokenizer(text)
    filtered_tokens = [word for word in tokenized_text if word not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_special_characters(text):
    '''removes special characters form the text'''
    tokenized_text = text_tokenizer(text)
    pattern = re.compile(r'[^a-zA-Z0-9 ]')
    filtered_tokens = list(filter(None, [pattern.sub(' ',token) for token in tokenized_text]))
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def tags_to_wordnet_tags(pos_tag):
    '''this function converts normal tags to wordnet tags'''
    if pos_tag.startswith('N'):
        return 'n'
    elif pos_tag.startswith('V'):
        return 'v'
    elif pos_tag.startswith('R'):
        return 'r'
    elif pos_tag.startswith('J'):
        return 'a'
    else:
        return None


def text_annotation(text):
    '''annotates the text with the universal tagset'''
    tokens = text_tokenizer(text)
    tagged_words = nltk.pos_tag(tokens)
    filtered_words = [(word, tags_to_wordnet_tags(tag)) for word, tag in tagged_words]
    return filtered_words


def text_lemmatization(text):
    '''returns the lemmatized text'''
    tagged_tokens = text_annotation(text)
    lemmatized_tokens = [wnl.lemmatize(word, tag) if tag else word for word, tag in tagged_tokens]
    filtered_text = ' '.join(lemmatized_tokens)
    return filtered_text


def normalizer(text):
    '''this function will completely normalize the text'''
    lower_case_text = text_lower_casing(text)
    filtered_stopwords = remove_stopwords(lower_case_text)
    filtered_special_characters = remove_special_characters(filtered_stopwords)
    lemmatized_text = text_lemmatization(filtered_special_characters)
    return lemmatized_text

