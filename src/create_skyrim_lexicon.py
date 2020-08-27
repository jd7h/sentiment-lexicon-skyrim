# Script for creating the Skyrim-specific sentiment lexicon
# judith van Stegeren

# this script assumes there is a frequency list of Skyrim words present in 
# skyrim_freq_[timestamp].json. You can create this list with create_skyrim_word_list.py.

import random
import csv
import json
import gensim 
import logging
import datetime
from clean_imperial_library import create_mapping, custom_stem

# calculate the PAD dimensions for one Skyrimword
def calculate_skyrim_PAD(word, model, eanewset, eanew, verbose=False):
    logging.debug("Calculating PAD values for {}".format(word))
    # find neighbours
    # probabilities = [(w, model.wv.similarity(w1=word, w2=w)) for w in eanewset] # doesn't work, we need error handling
    probabilities = []
    for w in eanewset:
        try:
            prob = model.wv.similarity(w1=word, w2=w)
        except KeyError as e:
            #print(e, type(e))
            prob = 0
        probabilities.append((w, prob))            
    probabilities.sort(key=lambda x:x[1], reverse=True)
    # calculate PAD values
    # average the VAD-values for each ANEW neighbour to calculate the VAD-value for the skyrim word
    val = 0
    ar = 0
    dom = 0
    logging.debug("Closest E-ANEW neighbours: {}".format(probabilities[:3]))
    for neighbour, prob in probabilities[:3]:
        val += eanew[neighbour]["valence"]
        ar += eanew[neighbour]["arousal"]
        dom += eanew[neighbour]["dominance"]
        if verbose:
            print("{} & {} & {} & {} \\\\".format(neighbour, eanew[neighbour]["valence"], eanew[neighbour]["arousal"], eanew[neighbour]["dominance"]))
    val = val / 3
    ar = ar / 3
    dom = dom / 3
    if verbose:
        print("{} & {} & {} & {} \\\\".format(word, val, ar, dom))
    return {"valence" : val, "arousal" : ar, "dominance" : dom}

# help function for calibrate_model
# pick TESTSETSIZE random words, calculate sentiment rating
# and check whether this sentiment rating is satisfactory/acceptable
# given the STD from EANEW
def validate_model_with_words(TESTSETSIZE, intersection, model, eanew):

    # setup
    eanewset = set(eanew.keys())
    dimensions = ['valence','arousal','dominance']
    good = {}
    for dim in dimensions:
        good[dim] = 0

    
    # remove words that have a different meaning in Skyrim
    blacklist = ["shout","cat","empire", "dominion"]
    for word in blacklist:
        if word in intersection:
            intersection.remove(word)

    # choose TESTSETSIZE random words to use for testing
    for i in range(TESTSETSIZE):
        testword = random.choice(intersection)
        logging.debug("testword: {}".format(testword))
        PAD = calculate_skyrim_PAD(testword, model, eanewset, eanew)
        for dimension in PAD:
            acceptable_min = eanew[testword][dimension] - eanew[testword][dimension+"_SD"]
            acceptable_max = eanew[testword][dimension] + eanew[testword][dimension+"_SD"]
            if PAD[dimension] > acceptable_min and PAD[dimension] < acceptable_max:
                logging.debug("{} has an acceptable PAD rating between {} and {}".format(testword,acceptable_min,acceptable_max))
                good[dimension] += 1
    
    logging.info("Results of calibration: {}".format(good))
    more_training_is_needed = False
    for dimension in dimensions:
        if good[dimension] < TESTSETSIZE:
            more_training_is_needed = True

    return more_training_is_needed

# input: unvalidated model
# output: model that has been retrained until at least 2 * TESTSETSIZE words
# from the intersection between E-ANEW and TES have received a sentiment rating
# within the standard deviation + mean of E-ANEW's human ratings
def calibrate(model, documents, eanew, intersection, TESTSETSIZE):
    ADDITIONAL_EPOCHS = 10 # nr of extra training rounds if model should be trained more
    
    more_training_is_needed = validate_model_with_words(TESTSETSIZE, intersection, model, eanew)
    if more_training_is_needed:
        model.train(documents,total_examples=len(documents),epochs=ADDITIONAL_EPOCHS)
        model, n = calibrate(model, documents, eanew, intersection, TESTSETSIZE)
        return model, n+ADDITIONAL_EPOCHS

    # double check for overfitting
    more_training_is_needed = validate_model_with_words(TESTSETSIZE, intersection, model, eanew)
    if more_training_is_needed:
        return calibrate(model, documents, eanew, intersection, TESTSETSIZE)

    return model, 0

def main(
        rating_warr_csv,
        filename_freq,
        filename_intersection,
        filename_lore,
        filename_stemmer,
        testsetsize,
        lastmodel,
        loadmodel=False
    ):
    """Assumptions:
    lorefile has been stemmed with a normal stemmer
    lorefile is lowercase
    lorefile is tokenizable with .tokenize

    """

    # load E-ANEW
    logging.info("Loading E-ANEW words from {}".format(rating_warr_csv))
    eanew_all = []
    with open(rating_warr_csv, newline='') as csvfile: # data/Ratings_Warriner_et_al.csv
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            eanew_all.append(row)
    # for each word, get the average valence, arousal and dominance rating
    # all fields: eanew_all[0].keys()
    # dict_keys(['Id', 'Word', 'V.Mean.Sum', 'V.SD.Sum', 'V.Rat.Sum', 'A.Mean.Sum', 'A.SD.Sum', 'A.Rat.Sum', 'D.Mean.Sum', 'D.SD.Sum', 'D.Rat.Sum', 'V.Mean.M', 'V.SD.M', 'V.Rat.M', 'V.Mean.F', 'V.SD.F', 'V.Rat.F', 'A.Mean.M', 'A.SD.M', 'A.Rat.M', 'A.Mean.F', 'A.SD.F', 'A.Rat.F', 'D.Mean.M', 'D.SD.M', 'D.Rat.M', 'D.Mean.F', 'D.SD.F', 'D.Rat.F', 'V.Mean.Y', 'V.SD.Y', 'V.Rat.Y', 'V.Mean.O', 'V.SD.O', 'V.Rat.O', 'A.Mean.Y', 'A.SD.Y', 'A.Rat.Y', 'A.Mean.O', 'A.SD.O', 'A.Rat.O', 'D.Mean.Y', 'D.SD.Y', 'D.Rat.Y', 'D.Mean.O', 'D.SD.O', 'D.Rat.O', 'V.Mean.L', 'V.SD.L', 'V.Rat.L', 'V.Mean.H', 'V.SD.H', 'V.Rat.H', 'A.Mean.L', 'A.SD.L', 'A.Rat.L', 'A.Mean.H', 'A.SD.H', 'A.Rat.H', 'D.Mean.L', 'D.SD.L', 'D.Rat.L', 'D.Mean.H', 'D.SD.H', 'D.Rat.H'])

    eanew = {}
    for row in eanew_all:
        word = row["Word"]
        eanew[word] = {}
        eanew[word]["valence"] = float(row["V.Mean.Sum"])
        eanew[word]["arousal"] = float(row["A.Mean.Sum"])
        eanew[word]["dominance"] = float(row["D.Mean.Sum"])
        # add Standard Deviations for calibrating model
        eanew[word]["valence_SD"] = float(row["V.SD.Sum"])
        eanew[word]["arousal_SD"] = float(row["A.SD.Sum"])
        eanew[word]["dominance_SD"] = float(row["D.SD.Sum"])

    # load TES, i.e. a list of skyrim specific words and their frequency in the lore
    logging.info("Loading TES frequency list from {}".format(filename_freq))
    skyrim_freq = []
    with open(filename_freq, "r") as skyrim_lexicon_file: # data/out/skyrim_freq[date].json
        skyrim_freq = json.loads(skyrim_lexicon_file.read())

    logging.info("Loading TES lore from {}".format(filename_lore))
    lore = []
    with open(filename_lore, "r") as lorefile:  # data/out/stemmed_library_[date].txt
        lore = lorefile.readlines()
        
    skyrim_words = set([w[0] for w in skyrim_freq])
    eanewset = set(eanew.keys())

    logging.info("Loading intersection from {}".format(filename_intersection))
    with open(filename_intersection, "r") as infile: # data/out/intersection.json
        intersection = json.loads(infile.read())
    # check whether our intersection creation went okay
    badwords = [word for word in intersection if word not in eanewset]
    if badwords:
        raise ValueError("The intersection contains words that are not part of E-ANEW: {}".format(badwords))

    # train word2vec on the TES lore with the parameters from the paper
    # Therese Bergsma used this blogpost as tutorial: https://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/

    if loadmodel:
        logging.info("Loading an earlier model: {}".format(lastmodel))
        # loading an earlier word2vec model and do not train a new one
        model = gensim.models.Word2Vec.load(lastmodel)
    else:
        logging.info("Training a new word2vec model on lore...")
        documents = []
        skyrim_stemmer = create_mapping(filename_stemmer)
        for line in lore:
            documents.append(gensim.utils.simple_preprocess(custom_stem(skyrim_stemmer, line)))
            logging.debug("Added book: {}".format(documents[-1][:50]))

        # create gensim word2vec model with specified parameters
        model = gensim.models.Word2Vec (documents, size=125, window=8, min_count=10, workers=10)
        model.train(documents,total_examples=len(documents),epochs=10)
        
        # calibrate the model on 20 words that are both in E-ANEW and the skyrimwords
        starttime = datetime.datetime.now()
        model, n = calibrate(model, documents, eanew, intersection, TESTSETSIZE=testsetsize)
        endtime = datetime.datetime.now()
        logging.info("model calibration took {}".format(endtime - starttime))
        logging.info("Total training epochs: {}".format(10 + n))
        # print some test statistics
        test_model(model)

        # save the word2vec model
        filename_w2vmodel = "data/out/skyrim_word2vec_" + datetime.datetime.strftime(datetime.datetime.now(),"%Y%m%d%H%M") + ".model"
        model.save(filename_w2vmodel)

    # for each skyrim-specific word, calculate a VAD-rating
    # for each skyrim-specific word, calculate the three E-ANEW words with the highest probability
    logging.info("Calculating PAD-values for TES words...")
    skyrim_lex = {}
    for word in skyrim_words:
        skyrim_lex[word] = calculate_skyrim_PAD(word, model, eanewset, eanew)
        logging.debug("PAD values for {}: {}".format(word, skyrim_lex[word]))
    
    # write the skyrim-lexicon (TES only) to a file of the form skyrim_lexicon_[currentdate].json
    filename_lexicon = "data/out/skyrim_lexicon_" + datetime.datetime.strftime(datetime.datetime.now(),"%Y%m%d%H%M") + ".json"
    logging.info("Writing TES lexicon to file: {}".format(filename_lexicon))
    with open(filename_lexicon, "w") as outfile:
        outfile.write(json.dumps(skyrim_lex))

    return filename_lexicon, model
    
# print a quick analysis of a created model
# show the most similar words for
# skyrim, altmer, nirnroot, dunmer, vivec, sheogorath
def test_model(model):
    for testword in "skyrim altmer nirnroot dunmer vivec sheogorath".split():
        print(testword)
        print("====================================")
        neighbours = model.wv.most_similar(testword)
        for n in neighbours:
            print(n)
        print()

def test_skyrim_lexicon():
    # loading an earlier model
    model = gensim.models.Word2Vec.load(lastmodel)
    test_model(model)

