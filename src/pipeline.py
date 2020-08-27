import logging
import csv
import create_skyrim_word_list
import create_skyrim_lexicon
import evaluate_lexicon

EANEW = "data/Ratings_Warriner_et_al.csv"
LORE = "data/out/stemmed_library_202003251211.txt"

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', filename="skyrim.log", level=logging.DEBUG)
    logging.info("------------------------Start pipeline----------------")

    # writes a wordlist with skyrim-specific words to filename_freq in the format "skyrim_freq_[currentdate].json"
    word_freq, word_freq_list, skyrim_stemmer, filename_freq, filename_stemmer, filename_intersection = create_skyrim_word_list.main(
        lorefile= LORE,
        rating_warr_csv=EANEW,
        filename_dict="data/dict_words_en.json"
        )

    logging.info("Creating Skyrim-specific lexicon")
    # writes a sentiment-lexicon to filename_lexicon in the format skyrim_lexicon_[currentdate].json
    filename_lexicon, model = create_skyrim_lexicon.main(
        rating_warr_csv = EANEW,
        filename_freq = filename_freq,
        filename_intersection = filename_intersection,
        filename_lore = LORE,
        testsetsize = 15,
        filename_stemmer = filename_stemmer,
        lastmodel = "",
        )

    logging.info("Running experiments: comparing lexicons to human ratings")
    # run main experiment: compare the lexicons E-ANEW, E-ANEW-TES and TES with the human ratings
    # print results (correlation metrics) of main experiment afterwards
    lexicons, evaluation_segments, evaluation_dialogues, human_ratings, human_ratings_stats, results, satisfactory, correlation = evaluate_lexicon.main(
        filename_evaluation_segments = "data/evaluation_dialogues.csv",
        filename_eanew_lexicon = EANEW,
        filename_gold_standard = "data/human_ratings_skyrim_dialogue.csv",
        filename_skyrim_lexicon = filename_lexicon,
        filename_stemmer = filename_stemmer
        )
        
    return satisfactory, correlation  
        
        
