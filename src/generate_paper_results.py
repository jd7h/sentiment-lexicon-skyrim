# generate results in latex for in paper

import logging
import gensim
import csv
import json
import pprint
import en_core_web_sm
from create_skyrim_word_list import main as create_freqlist
from create_skyrim_lexicon import calculate_skyrim_PAD
from evaluate_lexicon import load_lexicons, sentiment_rating
from evaluate_lexicon import main as evaluate
from clean_imperial_library import tokenize, create_mapping, clean_and_custom_stem_

#after experiments in pipeline are finished

filename_model = "data/out/skyrim_word2vec_202004161350.model"
rating_warr_csv = "data/Ratings_Warriner_et_al.csv"
dialoguefile = "data/skyrim_dialogue.txt"
filename_skyrim_lexicon = "data/out/skyrim_lexicon_202004161353.json"
stemmer_file = "data/out/skyrim_stemmer_202004161348.json"
filename_evaluation_segments = "data/evaluation_dialogues.csv"
filename_gold_standard = "data/human_ratings_skyrim_dialogue.csv"
stemmer_mapping = create_mapping(stemmer_file)
nlp = en_core_web_sm.load()
english_dictionary = "data/dict_words_en.json"
lorefile = "data/out/stemmed_library_202003251211.txt"

logging.basicConfig(logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', filename="skyrim.log", level=logging.DEBUG)


def sentiment_rating_latex(text, lexicon1, lexicon2, terminal=False):
    # assumption: there is no intersection between lexicon1.keys() and lexicon2.keys():
    output = []
    for word in tokenize(text):
        if clean_and_custom_stem_(word, stemmer_mapping, nlp) in lexicon1.keys():
            output.append(r"\textbf{{{}}}".format(word))
        elif clean_and_custom_stem_(word, stemmer_mapping, nlp) in lexicon2.keys():
            output.append(r"\underline{{{}}}".format(word))
        else:
            output.append(word)
    return " ".join(output)


# version that also propagates the original text (for latex printing)
def sentiment_rating_latex_(text, lexicon1, lexicon2):
    output = []
    lemma_and_original = clean_and_custom_stem_(text, stemmer_mapping, nlp, include_original = True)
    for lemma_and_ori in lemma_and_original:
        word = lemma_and_ori[0]
        ori = lemma_and_ori[1]
        if clean_and_custom_stem_(word, stemmer_mapping, nlp) in lexicon1.keys():
            output.append(r"\textbf{{{}}}".format(ori))
        elif clean_and_custom_stem_(word, stemmer_mapping, nlp) in lexicon2.keys():
            output.append(r"\underline{{{}}}".format(ori))
        else:
            output.append(ori)
    output = " ".join(output)
    output, n = re.subn(" \.",".",output)
    output, n = re.subn(" ,",",",output)
    output, n = re.subn(r"([A-Za-z{}]+) ?'nt",r"\1n't",output)
    output, n = re.subn(r"([A-Za-z{}]+) ?'m ",r"\1'm ",output)
    output, n = re.subn(r"([A-Za-z{}]+) ?'s ",r"\1's ",output)
    output, n = re.subn(r"([A-Za-z{}]+) ?'re ",r"\1're ",output)
    return output


# statistics about word list
create_freqlist(lorefile, rating_warr_csv, english_dictionary, output = False)

# 4.4 example calculating PAD values
example_word = "septim"

# load w2v model and return closest words to septim
model = gensim.models.Word2Vec.load(filename_model)
for word in model.wv.most_similar("septim"):
    print(word[0] + " & " + str(word[1]) + "\\\\")

logging.info("Loading E-ANEW words from {}".format(rating_warr_csv))
eanew = {}
with open(rating_warr_csv, newline='') as csvfile: # data/Ratings_Warriner_et_al.csv
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        word = row["Word"]
        eanew[word] = {}
        eanew[word]["valence"] = float(row["V.Mean.Sum"])
        eanew[word]["arousal"] = float(row["A.Mean.Sum"])
        eanew[word]["dominance"] = float(row["D.Mean.Sum"])
eanewset = set(eanew.keys())
print(len(eanew))
    
calculate_skyrim_PAD(example_word, model, eanewset, eanew, verbose=True)

# find a dialogue line with example word
dialogue = []
with open(dialoguefile, newline='') as csvfile:
    fieldnames = ['conv_id', 'quest_id', 'branch_id', 'topic', 'subtype', 'id', 'text', 'va_notes']
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames = fieldnames)
    for row in reader:
        dialogue.append(row)

for line in[row['text'] for row in dialogue if example_word in row['text']]:
    print(line)
    print()

example_line = "In the year 3E 41, Emperor Pelagius Septim was murdered in the Temple of the One in the Imperial City. Cut down by a Dark Brotherhood assassin."

# calculate sentiment rating for example line
# load lexicons
eanew, tes, combination = load_lexicons(rating_warr_csv,filename_skyrim_lexicon)

for lexicon in [eanew, tes, combination]:
    sentiment_rating(lexicon, example_line, stemmer_file, verbose = True)

print(sentiment_rating_latex_(example_line, eanew, tes))

# satisfactory results and correlation table in latex
lexicons, evaluation_segments, evaluation_dialogues, human_ratings, human_ratings_stats, results, satisfactory, correlation = evaluate(
    filename_evaluation_segments = filename_evaluation_segments,
    filename_eanew_lexicon = rating_warr_csv,
    filename_gold_standard = filename_gold_standard,
    filename_skyrim_lexicon = filename_skyrim_lexicon,
    filename_stemmer = stemmer_file
    )
    
# satisfactory looks like this: satisfactory[lexicon[1]][dimension]["dialogues"] = 0
output = ""
output  += r'\begin{tabular}{llll}'
output  += "\n"
output  += r' & & \multicolumn{2}{c}{Number of satisfactory ratings}\\'
output  += "\n"
output  += r' & dimension & E-ANEW & \extname{} \\'
output  += "\n"
for type_name in ['segments', 'dialogues']:
    output  += type_name
    for dim in ['valence', 'arousal', 'dominance']:
            output += r' & {} & {} & {} \\'.format(dim, satisfactory["E-ANEW"][dim][type_name], satisfactory["E-ANEW-TES"][dim][type_name])
            output += "\n"     

output += r'\end{tabular}'
print(output)
print()

# correlation table v1
# correlation looks like this: correlation_results["E-ANEW"][type_name][dim]["spearman"]

# pearson
# =========
output = ""
output  += r'\begin{tabular}{llrrrr}'
for metric in ['pearson', 'spearman']:
    output  += r'\toprule'
    output  += "\n"
    output  += r' & & \multicolumn{2}{c}{E-ANEW} & \multicolumn{2}{c}{\extname{}}\\\cmidrule(r){3-4}\cmidrule(l){5-6}'
    output  += "\n"
    output  += r' & & {} && {}\\'.format(metric.capitalize(), metric.capitalize())
    output  += "\n"
    output  += r' & & correlation & Significance & correlation & Significance \\'
    output  += "\n"
    for type_name in ['segments', 'dialogues']:
        output += r"\midrule"
        output += "\n"   
        output  += r'\multirow{{3}}{{*}}{{{}}}'.format(type_name.capitalize())
        for dim in ['valence', 'arousal', 'dominance']:
                if correlation["E-ANEW"][type_name][dim][metric] > correlation["E-ANEW-TES"][type_name][dim][metric]:
                    output += r' & {} & \textbf{{{:.4f}}} & {:.4f} & {:.4f} & {:.4f} \\'.format(dim.capitalize(), correlation["E-ANEW"][type_name][dim][metric], correlation["E-ANEW"][type_name][dim][metric + "_p"], correlation["E-ANEW-TES"][type_name][dim][metric], correlation["E-ANEW-TES"][type_name][dim][metric + "_p"])
                elif correlation["E-ANEW"][type_name][dim][metric] < correlation["E-ANEW-TES"][type_name][dim][metric]:
                    output += r' & {} & {:.4f} & {:.4f} & \textbf{{{:.4f}}} & {:.4f} \\'.format(dim.capitalize(), correlation["E-ANEW"][type_name][dim][metric], correlation["E-ANEW"][type_name][dim][metric + "_p"], correlation["E-ANEW-TES"][type_name][dim][metric], correlation["E-ANEW-TES"][type_name][dim][metric + "_p"])
                else:
                    output += r' & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\'.format(dim.capitalize(), correlation["E-ANEW"][type_name][dim][metric], correlation["E-ANEW"][type_name][dim][metric + "_p"], correlation["E-ANEW-TES"][type_name][dim][metric], correlation["E-ANEW-TES"][type_name][dim][metric + "_p"])
        output += "\n"
    output += r"\bottomrule"
    output += "\n"
    if metric == 'pearson':
        output += r'\\'
        output += "\n"
        output += r'\\'
        output += "\n"

output += r'\end{tabular}'

print(output)


# spearman
# =========
output = ""
output  += r'\begin{tabular}{llllll}'
output  += "\n"
output  += r' & & \multicolumn{2}{c}{E-ANEW} & \multicolumn{2}{c}{\extname{}}\\'
output  += "\n"
output  += r' & dimension & correlation & significance & correlation & significance \\'
output  += "\n"
for type_name in ['segments', 'dialogues']:
    output  += type_name
    for dim in ['valence', 'arousal', 'dominance']:
            if correlation["E-ANEW"][type_name][dim]["spearman"] > correlation["E-ANEW-TES"][type_name][dim]["spearman"]:
                output += r' & {} & \textbf{{{:.4f}}} & {:.4f} & {:.4f} & {:.4f} \\'.format(dim, correlation["E-ANEW"][type_name][dim]["spearman"], correlation["E-ANEW"][type_name][dim]["spearman_p"], correlation["E-ANEW-TES"][type_name][dim]["spearman"], correlation["E-ANEW-TES"][type_name][dim]["spearman_p"])
            elif correlation["E-ANEW"][type_name][dim]["spearman"] < correlation["E-ANEW-TES"][type_name][dim]["spearman"]:
                output += r' & {} & {:.4f} & {:.4f} & \textbf{{{:.4f}}} & {:.4f} \\'.format(dim, correlation["E-ANEW"][type_name][dim]["spearman"], correlation["E-ANEW"][type_name][dim]["spearman_p"], correlation["E-ANEW-TES"][type_name][dim]["spearman"], correlation["E-ANEW-TES"][type_name][dim]["spearman_p"])
            else:
                output += r' & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\'.format(dim, correlation["E-ANEW"][type_name][dim]["spearman"], correlation["E-ANEW"][type_name][dim]["spearman_p"], correlation["E-ANEW-TES"][type_name][dim]["spearman"], correlation["E-ANEW-TES"][type_name][dim]["spearman_p"])
            output += "\n"     

output += r'\end{tabular}'
print(output)
print()

# correlation table v2, Pearson & Spearman
# beetje te groot, zelfs fullwidth
# correlation looks like this: correlation_results["E-ANEW"][type_name][dim]["spearman"]
output = ""
output  += r'\begin{tabular}{llllllllll}\toprule'
output  += "\n"
output  += r' & & \multicolumn{4}{c}{E-ANEW} & \multicolumn{4}{c}{\extname{}}\\'
output  += "\n"
output  += r' & dimension & Pearson correlation & significance & Spearman correlation & significance & Pearson correlation & significance & Spearman correlation \\\midrule'
output  += "\n"
for type_name in ['segments', 'dialogues']:
    output  += type_name
    for dim in ['valence', 'arousal', 'dominance']:
        output  += r' & {} '.format(dim)
        for lexicon in ['E-ANEW', 'E-ANEW-TES']:
            if correlation[lexicon][type_name][dim]['pearson'] > correlation[lexicon][type_name][dim]['spearman']:
                highest = 'pearson'
            elif correlation[lexicon][type_name][dim]['pearson'] < correlation[lexicon][type_name][dim]['spearman']:
                highest = 'spearman'
            else:
                highest = ""
            for metric in ['pearson', 'spearman']:
                if metric == highest:
                    output += r'& \textbf{{{:.6f}}} & {:.6f}'.format(correlation[lexicon][type_name][dim][metric], correlation[lexicon][type_name][dim][metric + "_p"])
                else:
                    output += r'& {:.6f} & {:.6f}'.format(correlation[lexicon][type_name][dim][metric], correlation[lexicon][type_name][dim][metric + "_p"])
                
        output += r'\\'
        output += "\n"


output += r'\end{tabular}'
print(output)
print()

# load evaluation dialogues
logging.info("Loading evaluation sentences from {}".format(filename_evaluation_segments[:-4] + "_interpunction.csv"))
unprocessed_evaluation_segments = []
with open(filename_evaluation_segments[:-4] + "_interpunction.csv","r") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        unprocessed_evaluation_segments.append(row)


# printing table in appendix with segments and dialogues and recognized words
output = ""
output  += r'\begin{tabular}{llp{13cm}}\toprule'
output  += "\n"
output  += r'Dialogue & Segment & Text\\\midrule'
output  += "\n"
for index, segment in enumerate(unprocessed_evaluation_segments):
    if index % 4 == 0:
        output += r'\multirow{{4}}{{*}}{{{}}} '.format(segment['dialogue_id'])
    output += r'& {} & {}\\'.format(segment['segment_id'], sentiment_rating_latex_(segment['text'],eanew,tes))
    if index % 4 == 3:
        output += r'\midrule'
    if index == len(unprocessed_evaluation_segments)-1:
        output += r'\bottomrule'
    output += "\n"

output  += r'\end{tabular}'
output, n = re.subn(" \.",".",output)
output, n = re.subn(" ,",",",output)
output, n = re.subn(r"([A-Za-z{}]+) ?'nt",r"\1n't",output)
output, n = re.subn(r"([A-Za-z{}]+) ?'m ",r"\1'm ",output)
output, n = re.subn(r"([A-Za-z{}]+) ?'s ",r"\1's ",output)
output, n = re.subn(r"([A-Za-z{}]+) ?'re ",r"\1're ",output)
print(output)
print("LET OP: MULTIROW PARAMETERS MOETEN MET DE HAND WORDEN GEREPAREERD")

# plotting distribution of human ratings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

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
        human_ratings_stats[s][dim]["min"] = min([participant[s][dim] for participant in huf, ax = plt.subplots(figsize=(6, 6))
man_ratings_])
        human_ratings_stats[s][dim]["max"] = max([participant[s][dim] for participant in human_ratings_])
        
# print metrics for normal distribution


# tryout: data = [participant[segment]['valence'] for participant in human_ratings_ for segment in participant.keys()]


# plotting data for all three dimensions and both text types
data_min = 1
data_max = 9
f, ax = plt.subplots(2, 3, figsize=(6, 6)) # make 3 subplots, landscape format
#sns.distplot(d, kde=False, color="b", ax=axes[0])
bins = np.arange(data_min, data_max + 1.5) - 0.5
for ploty, texttype in enumerate(['s', 'd']):
    for plotx, dimension in enumerate(['valence', 'arousal', 'dominance']):
        data = [participant[segment][dimension] for participant in human_ratings_ for segment in participant.keys() if texttype in segment]
        sns.distplot(data, bins=bins, color="b", ax=ax[ploty, plotx])
        if texttype == "s":
            ax[ploty, plotx].set_title('{} ratings for {}'.format(dimension.capitalize(), "segments"))
            ax[ploty, plotx].set_ylabel('number of segments')
            #ax[ploty, plotx].set_yticks(np.arange(0,80,10))
        else:
            ax[ploty, plotx].set_title('{} ratings for {}'.format(dimension.capitalize(), "complete dialogues"))
            ax[ploty, plotx].set_ylabel('number of dialogues')   
            #ax[ploty, plotx].set_yticks(np.arange(0,30,5)) 
        ax[ploty, plotx].set_xlabel('sentiment rating')
        ax[ploty, plotx].set_xticks(bins[:-1]+0.5)
        #kde_ax = ax[ploty, plotx].twinx()  # instantiate a second axes that shares the same x-axis
        #sns.kdeplot(data=data, ax=kde_ax, bw=1)

    
f.show()

