import glob
import os
import pathlib

import nltk
import string
import unidecode
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk import ngrams


def main():
    raw_data_dir_path = "data"
    results_dir_path = "results"
    GRAM_SIZE = 1
    TOP_X_WORDS = 30
    base_len = {}

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)

    for file_path in glob.glob(raw_data_dir_path + "/*.csv"):
        data_frame = pd.read_csv(file_path)
        data_frame = data_frame[data_frame["saida"] == 0]
        file_name = os.path.basename(file_path)
        file_name = file_name.replace(pathlib.Path(file_name).suffix, "")

        # Quantidade de entradas
        data_frame_len = len(data_frame.index)
        base_len[file_name] = data_frame_len

        words_freq = {}
        for _, curr_row in data_frame.iterrows():
            curr_row_content = unidecode.unidecode(curr_row["conteudo"].lower().strip())
            curr_row_words_filtered = filter(
                lambda word: word not in stopwords.words("portuguese") and word not in string.punctuation,
                curr_row_content.split())
            curr_row_words_filtered = ngrams(curr_row_words_filtered, GRAM_SIZE)
            curr_row_words_filtered = [' '.join(grams) for grams in curr_row_words_filtered]
            

            for word in curr_row_words_filtered:
                if word in words_freq:
                    words_freq[word] += 1
                else:
                    words_freq[word] = 1

        # Distribuição de frequência
        words_freq_sorted = sorted(words_freq.items(), key=lambda kv: kv[1], reverse=True)[0:TOP_X_WORDS]
        x = [i[0] for i in words_freq_sorted]
        y = [i[1] for i in words_freq_sorted]
        plt.clf()
        plt.barh(x, y)
        plt.xlabel("Quantidade de Ocorrências")
        plt.savefig(results_dir_path + "/" + file_name + "_freq_dist.png", bbox_inches="tight")

        # Nuvem de palavras
        word_cloud = WordCloud(width=800, height=800, background_color="white", min_font_size=10,
                               collocations=False).generate_from_frequencies(dict(words_freq_sorted))
        plt.clf()
        plt.imshow(word_cloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(results_dir_path + "/" + file_name + "_word_cloud.png", bbox_inches="tight")

        # Salvando dados da análise
        file = open(results_dir_path + "/" + file_name + "_analysis.txt", "w")
        file.write("Quantidade de entradas: " + str(data_frame_len) + "\n")
        file.write("Distribuição de palavras ("+str(TOP_X_WORDS)+" mais frequentes): " + str(words_freq_sorted) + "\n")
        file.close()

    #Salvando gráfico com a quantidade de entradas em cada base
    plt.clf()
    plt.bar(base_len.keys(), base_len.values())
    plt.xlabel("Quantidade de Entradas")
    plt.savefig(results_dir_path + "/" + "base_qtd_entries.png", bbox_inches="tight")


if __name__ == '__main__':
    main()
