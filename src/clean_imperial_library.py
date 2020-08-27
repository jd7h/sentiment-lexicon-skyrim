import json
import logging
import nltk
import re
import spacy
import en_core_web_sm
import datetime

def custom_stem(mapping, text):    
    newtokens = []
    for token in tokenize(text):
        if token in mapping.keys():
            newtokens.append(mapping[token])
        else:
            newtokens.append(token)
    return " ".join(newtokens)
    
def create_mapping(sourcefile):
    with open(sourcefile,"r") as infile: # data/out/stemmer[date].json
        stemmer = json.loads(infile.read())
    mapping = {}
    for singular in stemmer.keys():
        for variation in stemmer[singular]:
            mapping[variation] = singular
    return mapping

def tokenize(text):
    return text.split()

def has_alpha(text):
    return re.search("[A-Za-z]", text)

def clean_and_stem(nlp, text, include_original = False):
    lines = text.split("\n") #split on newline as not all sentences end with period
    lines = [line.strip() for line in lines] # remove extra whitespace
    # remove -
    lines = [re.subn("-","",line)[0] for line in lines] # remove -
    # tokenize into sentences
    sentences = [sentence for line in lines for sentence in nltk.sent_tokenize(line)] # use nltk sentence tokenizer
    if include_original:
        return [(token.lemma_.lower(), token.text) for sentence in sentences for token in nlp(sentence)]
    return " ".join([token.lemma_.lower() for sentence in sentences for token in nlp(sentence) if has_alpha(token.lemma_) and token.lemma_ != "-PRON-"])

# perfectly fine to use this one, but it's slow
def clean_and_custom_stem(text, stemmer_file):
    mapping = create_mapping(stemmer_file)
    nlp = en_core_web_sm.load()
    return custom_stem(mapping, clean_and_stem(nlp, text.lower()))

# faster implementation with resource preloading    
def clean_and_custom_stem_(text, mapping, nlp, include_original = False):
    if include_original:
        lemma_and_original_list = clean_and_stem(nlp, text, include_original=include_original)
        return [(custom_stem(mapping, x[0]), x[1]) for x in lemma_and_original_list]
    return custom_stem(mapping, clean_and_stem(nlp, text.lower()))
    

def main(filename_library, book_filtering = False):
    # load library.json
    with open(filename_library,"r") as infile: # data/library.json
        data = json.loads(infile.read())

    if book_filtering:
        # filter out double books
        # use the longest version of each book
        titles = set([x['title'] for x in data])
        data_ = []
        for title in titles:
            versions = [book for book in data if book['title'] == title]
            longest_version = [book for book in versions if len(book['text']) == max([len(book['text']) for book in versions])][0]
            data_.append(longest_version)
        data = data_
        
    nlp = en_core_web_sm.load() # load the small webcorpus from spacy
    
    stemmed_books = []
    for book in data:
        stemmed_books.append(clean_and_stem(nlp, book['text']))
    
    with open("data/out/stemmed_library_" + datetime.datetime.strftime(datetime.datetime.now(),"%Y%m%d%H%M") + ".txt", "w") as outfile:
        for book in stemmed_books:
            outfile.write(book)
            outfile.write("\n")

