# Script for evaluating the Skyrim-specific sentiment lexicon as compared to human raters and E-ANEW.
import csv
import json
import logging
import warnings
import en_core_web_sm
from numpy import std, mean
from scipy.stats import pearsonr, spearmanr
from clean_imperial_library import tokenize, clean_and_custom_stem

def sentiment_rating(lexicon, text, stemmer_file, verbose = False):
    logging.debug("Calculating sentiment rating for {}".format(text))
    # assumption: lexicon contains PAD-values
    dimensions = ['valence','arousal','dominance']
    # assumption: lexicon keys are lowercase
    # initalize
    hits = []
    rating = {}
    for dimension in dimensions:
        rating[dimension] = 0
    # average PAD-values of all words in text that occur in our lexicon
    clean_text = clean_and_custom_stem(text, stemmer_file)
    for word in tokenize(clean_text):
        if word in lexicon.keys(): # alternative: if word in lexicon.keys() and not word in hits
            hits.append(word)
            for dimension in dimensions:
                rating[dimension] += lexicon[word][dimension]
    if len(hits) > 0:
        for dimension in dimensions:
            rating[dimension] = rating[dimension] / len(hits)
    logging.debug("Emotion words: {} hits: {}".format(len(hits), hits))
    logging.debug("Sentiment rating: {}".format(rating))
    logging.debug("===================================================")
    if verbose:
        print("Uncleaned text: {}".format(text))
        print("Cleaned text: {}".format(clean_text))
        print("Emotion words: {} hits: {}".format(len(hits), hits))
        print("Sentiment rating: {}".format(rating))
        print("Latex:")
        som = []
        for emotion_word in hits:
            som.append("({} : {:.2f})".format(emotion_word, lexicon[emotion_word]["valence"]))
        print("(" + " + ".join(som) + ") ", end="")
        print("/ {} = {:.2f}".format(len(hits), rating["valence"]))
        print("===================================================")
    return rating
    
def load_lexicons(filename_eanew_lexicon,filename_skyrim_lexicon):
    # load lexicons
    # E-ANEW
    logging.info("Loading E-ANEW words from {}".format(filename_eanew_lexicon))
    eanew = {}
    # load E-ANEW
    with open(filename_eanew_lexicon, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            word = row["Word"]
            eanew[word] = {}
            eanew[word]["valence"] = float(row["V.Mean.Sum"])
            eanew[word]["arousal"] = float(row["A.Mean.Sum"])
            eanew[word]["dominance"] = float(row["D.Mean.Sum"])

    # TES
    logging.info("Loading TES words from {}".format(filename_skyrim_lexicon))
    tes = {}
    with open(filename_skyrim_lexicon, newline='') as skyrim_lex:
        tes = json.loads(skyrim_lex.read())

    # E-ANEW-TES
    # we copy TES and add all words from E-ANEW
    combination = tes.copy()
    combination.update(eanew)

    return eanew, tes, combination
        
def main(
        filename_evaluation_segments,
        filename_eanew_lexicon,
        filename_gold_standard,
        filename_skyrim_lexicon,
        filename_stemmer
    ):
    # load evaluation segments
    logging.info("Loading evaluation sentences from {}".format(filename_evaluation_segments))
    evaluation_segments = []
    with open(filename_evaluation_segments,"r") as csvfile: # data/evaluation_dialogues.csv
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            evaluation_segments.append(row)

    # combine segments into dialogues
    evaluation_dialogues = [(dialogue, " ".join([segment["text"] for segment in evaluation_segments if segment["dialogue_id"] == dialogue])) 
        for dialogue in {segment["dialogue_id"] for segment in evaluation_segments}]

    # load lexicons
    eanew, tes, combination = load_lexicons(filename_eanew_lexicon,filename_skyrim_lexicon)
    lexicons = [(eanew, "E-ANEW"), (tes, "TES"), (combination, "E-ANEW-TES")]    

    # quick check to see whether all keys are indeed lowercase and have PAD-values:
    notlower = [key for key in list(combination.keys()) if not key.islower()]
    missingpad = [key for key in list(combination.keys()) \
                    if  not "arousal" in combination[key].keys() or \
                        not "valence" in combination[key].keys() or \
                        not "dominance" in combination[key].keys()]
    if notlower:
        warnings.warn("Not all emotion words are lowercase: {}".format(notlower))
    if missingpad:
        raise ValueError("Not all emotion words have PAD-values: {}".format(missingpad))

    # load gold standard: human ratings
    logging.info("Loading gold standard human ratings from {}".format(filename_gold_standard))
    human_ratings = []
    with open(filename_gold_standard,"r") as csvfile: #  "data/human_ratings_skyrim_dialogue.csv"
        reader = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        for row in reader:
            human_ratings.append(row)

    # rewrite gold standard to same format as lexicons
    human_ratings_ = []
    for p in human_ratings:
        participant = {}
        for key in human_ratings[0].keys():
            s_id, dimension = key.split("_")
            if s_id not in participant.keys():
                participant[s_id] = {}
            participant[s_id][dimension] = p[key]
        human_ratings_.append(participant)

    human_ratings_stats = {}
    for s in human_ratings_[0].keys():
        human_ratings_stats[s] = {}
        for dim in human_ratings_[0][s].keys():
            human_ratings_stats[s][dim] = {}
            human_ratings_stats[s][dim]["mean"] = mean([participant[s][dim] for participant in human_ratings_])  
            human_ratings_stats[s][dim]["std"] = std([participant[s][dim] for participant in human_ratings_])
            human_ratings_stats[s][dim]["min"] = min([participant[s][dim] for participant in human_ratings_])
            human_ratings_stats[s][dim]["max"] = max([participant[s][dim] for participant in human_ratings_])

    # for every dialogue, calculate a rating per segment and one for the dialogue as a whole
    # initialize
    results = {}
    for (lexicon, lex_name) in lexicons:
        results[lex_name] = {}

    for (lexicon, lex_name) in lexicons:
        logging.info("Lexicon {}".format(lex_name))
        # calculate for segments
        for segment in evaluation_segments:
            segment_id = segment["segment_id"]
            logging.info("Calculating sentiment score for segment {}: {}".format(segment_id, segment["text"][:50]))
            results[lex_name][segment_id] = sentiment_rating(lexicon, segment["text"], filename_stemmer)
        # calculate for dialogues
        for dialogue_id, text in evaluation_dialogues:
            logging.info("Calculating sentiment score for dialogue {}: {}".format(dialogue_id, text[:50]))
            results[lex_name][dialogue_id] = sentiment_rating(lexicon, text, filename_stemmer)

    # print the results of the experiments
    print()
    satisfactory = generate_paper_v1_results(lexicons, results, human_ratings_stats)
    print()
    correlation = generate_paper_v2_results(lexicons, results, human_ratings_, human_ratings_stats)

    return lexicons, evaluation_segments, evaluation_dialogues, human_ratings_, human_ratings_stats, results, satisfactory, correlation

# a sentiment rating from a lexicon is satisfactory if it falls in the range given by [human_rating_mean - SD, human_rating_mean + SD]
def is_satisfactory(lexicon_rating, human_rating, human_rating_SD):
    return lexicon_rating >= (human_rating - human_rating_SD) and lexicon_rating <= (human_rating + human_rating_SD)

# create a dictionary that lists whether, for all ratings by all lexicons, the ratings 
# are 'satisfactory' according to the definition in the paper
def generate_paper_v1_results(lexicons, results, human_ratings_stats):
    segments = [segment for segment in results["E-ANEW"].keys() if "s" in segment]
    dialogues = [segment for segment in results["E-ANEW"].keys() if "d" in segment]
    dimensions = ['valence','arousal','dominance']
    
    # init dict for results
    satisfactory = {}
    for lexicon in lexicons:
        satisfactory[lexicon[1]] = {}
        for dimension in dimensions:
            satisfactory[lexicon[1]][dimension] = {}
            satisfactory[lexicon[1]][dimension]["segments"] = 0
            satisfactory[lexicon[1]][dimension]["dialogues"] = 0
    
    for lexicon in lexicons:
        print(lexicon[1]) # print the name of the lexicon
        #Did the segments get a satisfactory rating?
        for segment in segments:
            for dimension in dimensions:
                satisfactory_segment = is_satisfactory(results[lexicon[1]][segment][dimension], human_ratings_stats[segment][dimension]["mean"], human_ratings_stats[segment][dimension]["std"])
                print(segment, dimension, satisfactory_segment)
                if satisfactory_segment:
                    satisfactory[lexicon[1]][dimension]['segments'] += 1
        #Did the dialogues get a satisfactory rating?
        for segment in dialogues:
            for dimension in dimensions:
                satisfactory_segment = is_satisfactory(results[lexicon[1]][segment][dimension], human_ratings_stats[segment][dimension]["mean"], human_ratings_stats[segment][dimension]["std"])
                print(segment, dimension, satisfactory_segment)
                if satisfactory_segment:
                    satisfactory[lexicon[1]][dimension]['dialogues'] += 1
    return satisfactory

def generate_paper_v2_results(lexicons, results, human_ratings, human_ratings_stats):
    segments = [segment for segment in results["E-ANEW"].keys() if "s" in segment]
    dialogues = [segment for segment in results["E-ANEW"].keys() if "d" in segment]
    dimensions = ['valence','arousal','dominance']
    
    correlation_results = {}

    print("Correlations between the gold standard and the various lexicons:")
    # correlatie tussen de lexicons en de gold standard:
    for lexicon, lexicon_name in lexicons:
        correlation_results[lexicon_name] = {}
        print("Pearson correlations between human ratings (gold standard) and", lexicon_name)
        for data_type, type_name in [(segments, "segments"), (dialogues, "dialogues")]:
            correlation_results[lexicon_name][type_name] = {}
            for dim in dimensions:
                res_hum = [human_ratings_stats[segment][dim]["mean"] for segment in data_type]
                res_lex = [results[lexicon_name][segment][dim] for segment in data_type]
                corr, pval = pearsonr(res_hum, res_lex)
                correlation_results[lexicon_name][type_name][dim] = {}
                correlation_results[lexicon_name][type_name][dim]["pearson"] = corr
                correlation_results[lexicon_name][type_name][dim]["pearson_p"] = pval
                print(type_name, dim, corr, pval)
        print()

    # correlatie tussen de lexicons en de gold standard:
    for lexicon, lexicon_name in lexicons:
        print("Spearman's Rho (non-parametric correlation) between human ratings (gold standard) and", lexicon_name)
        for data_type, type_name in [(segments, "segments"), (dialogues, "dialogues")]:
            for dim in dimensions:
                res_hum = [human_ratings_stats[segment][dim]["mean"] for segment in data_type]
                res_lex = [results[lexicon_name][segment][dim] for segment in data_type]
                corr, pval = spearmanr(res_hum, res_lex)
                correlation_results[lexicon_name][type_name][dim]["spearman"] = corr
                correlation_results[lexicon_name][type_name][dim]["spearman_p"] = pval
                print(type_name, dim, corr, pval)
        print()
    
    return correlation_results
