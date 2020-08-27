# judith van stegeren
# skyrim NLP
# 2020-02-19

import json
import datetime
import logging
import csv
from clean_imperial_library import tokenize

def main(lorefile, rating_warr_csv, filename_dict, output = True):
    """Assumption: 
    lorefile has been stemmed with a normal stemmer
    lorefile is lowercase
    lorefile is tokenizable with .tokenize
    """
    logging.info("Creating Skyrim wordlist")

    # load lore    
    logging.info("Loading uncleaned lorefile from {}".format(lorefile))
    with open(lorefile, "r") as loreinput:
	    lore = loreinput.read()

    logging.debug("Lore: {}".format(lore[:50]))

    # load E-ANEW
    logging.info("Loading E-ANEW words from {}".format(rating_warr_csv))
    with open(rating_warr_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        eanew = [row["Word"] for row in reader]
         
    logging.debug("E-ANEW words: {}...".format(", ".join(eanew)[:50]))

    # count words in lore
    word_freq = {}

    # initialize frequency list
    clean_lore = tokenize(lore.lower())
    clean_lore_set = set(clean_lore)
    for word in clean_lore_set:
        word_freq[word] = 0

    logging.info("Counting unique tokens in lore...")
    for word in clean_lore:
        word_freq[word] += 1
        
    # compute the intersection of the e-anew words and the words in lore
    logging.info("Computing intersection of TES words and E-ANEW words")
    intersection = [word for word in eanew if word in clean_lore_set and word_freq[word] >= 10]
    
    filename_intersection = "data/out/skyrim_intersection_" + datetime.datetime.strftime(datetime.datetime.now(),"%Y%m%d%H%M") + ".json"
    if output:
        logging.info("Writing skyrim and E-ANEW intersection to file: {}".format(filename_intersection))
        with open(filename_intersection, "w") as outfile:
            outfile.write(json.dumps(intersection))

    # remove english words
    # load english dict
    filename_dict
    logging.info("Loading English dictionary from {}".format(filename_dict))
    with open(filename_dict) as f:
	    dict_words = json.load(f)

    dict_set = set(dict_words)
    logging.info("Removing all English words from frequency list")
    # drop all words present in our english dictionary
    english_words_dropped = 0
    for word in set(word_freq.keys()):
        if word in dict_set or (type(word) == tuple and word[0] in dict_set):
            word_freq.pop(word)
            english_words_dropped += 1
            
    if "false" in word_freq.keys():
        word_freq.pop("false")
        english_words_dropped += 1

    logging.debug("Dropped {} English words from frequency list".format(english_words_dropped))
    logging.debug("Words left in frequency list: {}".format(len(word_freq.keys())))
    print("Words left in frequency list: {}".format(len(word_freq.keys())))

    # merge frequency of plurals with singular words
    # and save variations on words for later usage
    logging.info("Creating custom stemmer for TES lore")
    skyrim_stemmer = {}
    
    plurals = [word for word in word_freq.keys() if word[:-1] in word_freq.keys()]
    logging.debug("Plurals: {}".format(str(plurals)))
    plurals.sort()
    plurals.reverse()
    for plural in plurals:
        singular = plural[:-1]
        logging.debug("Singular / plural: {} / {}".format(singular, plural))
        if word_freq[singular] < 10: # if it occurs < 10 times it might be a typo, skip
            logging.debug("Pair occurs < 10 times, ignoring plural")
            continue
        if len(singular) < 5: # if the word[-1] occurs and it's very short, it's probably coincidence
            logging.debug("Singular is short, plural is probably coincidence, ignoring plural")
            continue 
        if singular not in skyrim_stemmer.keys():
            skyrim_stemmer[singular] = []
        skyrim_stemmer[singular].append(plural)
        word_freq[plural[:-1]] += word_freq[plural] # add frequency of plural form to freq of singular form
        word_freq.pop(plural)
        
    #merge all linked lists of stems
    stems_longest_first = list(set(skyrim_stemmer.keys()))
    stems_longest_first.sort(reverse=True, key=lambda x:len(x))
    for stem in stems_longest_first:
        if stem[:-1] in skyrim_stemmer.keys():
            for variant in skyrim_stemmer[stem]:
                skyrim_stemmer[stem[:-1]].append(variant)
            skyrim_stemmer.pop(stem)

    logging.info("Custom stemmer: {}".format(skyrim_stemmer.items()))
        
    # drop anything < 10 occurrences
    logging.info("Dropping all low-frequency words from frequency list")
    for key in set(word_freq.keys()):
        if word_freq[key] < 10:
            word_freq.pop(key)

    # make the dictionary (word, frequency) a list and sort it by frequency
    word_freq_list = list(word_freq.items())
    word_freq_list.sort(key=lambda x:x[1])
    logging.info("50 words with highest frequency: {}".format(word_freq_list[-50:]))
    logging.info("Length of frequency list: {} words".format(len(word_freq_list)))
    
    filename_freq = ""
    filename_stemmer = ""

    if output:
        # write results to file
        # word_freq_list.reverse()
        filename_freq = "data/out/skyrim_freq_" + datetime.datetime.strftime(datetime.datetime.now(),"%Y%m%d%H%M") + ".json"
        logging.info("Writing frequency list to file: {}".format(filename_freq))
        with open(filename_freq,"w") as outfile: 
            outfile.write(json.dumps(word_freq_list))
        
        filename_stemmer = "data/out/skyrim_stemmer_" + datetime.datetime.strftime(datetime.datetime.now(),"%Y%m%d%H%M") + ".json"
        logging.info("Writing skyrim stemmer to file: {}".format(filename_stemmer))
        with open(filename_stemmer,"w") as outfile:
            outfile.write(json.dumps(skyrim_stemmer))

    return word_freq, word_freq_list, skyrim_stemmer, filename_freq, filename_stemmer, filename_intersection
