import glob
import os
import pathlib
import csv
import string

import nltk
import unidecode
import spacy
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Rodar o comando "python -m spacy download pt" no terminal antes
nlp = spacy.load('pt_core_news_sm')

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def lemma_sentence(sentence):
    lemma = ""

    for token in nlp(sentence):
        lemma += token.lemma_ + " "

    return lemma


def stem_sentence(sentence):
    snow = SnowballStemmer(language='portuguese')
    token_words = word_tokenize(sentence)
    stem = []

    for word in token_words:
        stem.append(snow.stem(word))
        stem.append(" ")

    return "".join(stem)


def main():
    raw_data_dir_path = 'data'
    results_dir_path = 'results'
    lemma_results_dir_path = 'results_lemma'
    stem_results_dir_path = 'results_stem'
    csv_header = [['tag', 'conteudo', 'saida']]
    all_dfs = []

    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)

    if not os.path.exists(lemma_results_dir_path):
        os.makedirs(lemma_results_dir_path)

    if not os.path.exists(stem_results_dir_path):
        os.makedirs(stem_results_dir_path)

    for file_path in glob.glob(raw_data_dir_path + "/*.csv"):
        df = pd.read_csv(file_path)
        df_only_correct = df[df['saida'] == 0]
        file_name = os.path.basename(file_path)
        file_name_no_ext = file_name.replace(pathlib.Path(file_name).suffix, "")

        if file_name_no_ext == 'PROCESSO':
            process_sample = df.sample(n=16344)
            process_sample_only_correct = process_sample[process_sample['saida'] == 0]
            file_name_no_ext += "_sample"

            process_sample.to_csv(os.path.join(results_dir_path, file_name_no_ext + ".csv"), index=False)

            df = process_sample
            df_only_correct = process_sample_only_correct

        all_dfs.append(df)

        df_only_correct.to_csv(os.path.join(results_dir_path, file_name_no_ext + "_only_correct.csv"), index=False)

        lemma_csv_content = []
        lemma_only_correct_csv_content = []
        stem_csv_content = []
        stem_only_correct_csv_content = []
        for _, curr_row in df.iterrows():
            curr_row_content = unidecode.unidecode(curr_row['conteudo'].lower().strip())
            curr_row_words_filtered = filter(
                lambda word: word not in stopwords.words('portuguese') and word not in string.punctuation,
                curr_row_content.split())
            curr_sentence = " ".join(curr_row_words_filtered)
            lemma_result = lemma_sentence(curr_sentence)
            stem_result = stem_sentence(curr_sentence)
            lemma_csv_content.append([curr_row['tag'], lemma_result, curr_row['saida']])
            stem_csv_content.append([curr_row['tag'], stem_result, curr_row['saida']])

            if curr_row['saida'] == 0:
                lemma_only_correct_csv_content.append([curr_row['tag'], lemma_result, curr_row['saida']])
                stem_only_correct_csv_content.append([curr_row['tag'], stem_result, curr_row['saida']])

        with open(os.path.join(lemma_results_dir_path, file_name_no_ext + "_lemma.csv"), 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_header + lemma_csv_content)

        with open(os.path.join(stem_results_dir_path, file_name_no_ext + "_stem.csv"), 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_header + stem_csv_content)

        with open(os.path.join(lemma_results_dir_path, file_name_no_ext + "_lemma_only_correct.csv"), 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_header + lemma_only_correct_csv_content)

        with open(os.path.join(stem_results_dir_path, file_name_no_ext + "_stem_only_correct.csv"), 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_header + stem_only_correct_csv_content)

    all_dfs_df = pd.concat(all_dfs)
    all_dfs_df.to_csv(os.path.join(results_dir_path, "unified_data.csv"), index=False)


if __name__ == '__main__':
    main()
