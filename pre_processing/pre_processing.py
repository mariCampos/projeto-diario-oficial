import glob
import os
import pathlib
import csv

import nltk
import string
import unidecode
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer


def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(snow.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

# For Lemmatization
# python3 -m spacy download pt_core_news_sm
import spacy
nlp = spacy.load("pt_core_news_sm")

def lemmaSetence(setence):
    lemma = ""
    for token in nlp(setence):
        lemma += token.lemma_ + " "

    return lemma


GRAM_SIZE = 1
TOP_X_WORDS = 30
snow = SnowballStemmer(language="portuguese")
lemmatizer = WordNetLemmatizer() 
raw_data_dir_path = "data"
results_dir_path = "results_lemma"

if not os.path.exists(results_dir_path):
    os.makedirs(results_dir_path)

for file_path in glob.glob(raw_data_dir_path + "/*.csv"):
    data_frame = pd.read_csv(file_path)
    data_frame = data_frame[data_frame["saida"] == 0]
    file_name = os.path.basename(file_path)
    file_name = file_name.replace(pathlib.Path(file_name).suffix, "")

    result_path = file_path.replace('data', 'results')
    csv_data = [['tag', 'conteudo', 'saida']]

    for _, curr_row in data_frame.iterrows():
        curr_row_content = unidecode.unidecode(curr_row["conteudo"].lower().strip())
        curr_row_words_filtered = filter(
            lambda word: word not in stopwords.words("portuguese") and word not in string.punctuation,
            curr_row_content.split())
        curr_row_words_filtered = ngrams(curr_row_words_filtered, GRAM_SIZE)
        curr_row_words_filtered = [' '.join(grams) for grams in curr_row_words_filtered]
        new_current = lemmaSetence(' '.join(curr_row_words_filtered))
        csv_data.append([curr_row["tag"], new_current, curr_row["saida"]])

    #inserir na tabela especifica de resultados
    with open(result_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)

    


