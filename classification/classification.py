from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import pandas as pd
import naive_bayes as nb
import knn 
import numpy as np
import os
import glob
import pathlib


def build_args_parser():
    usage = 'python classification.py -o <recognition option>\n       ' \
            'run with --help for arguments descriptions'
    parser = ArgumentParser(description='An algorithms suite to classify Diário Oficial documents', usage=usage)

    parser.add_argument('-o', '--option', dest='option', type=int, default=1, choices=[1, 2, 3],
                        help='Type of classification algorithm. Could be:\n       '
                             '1 - Naive Bayes')

    return parser


def main():
    args_parser = build_args_parser()
    args = args_parser.parse_args()

    results_dir_path = 'results'
    raw_data_dir_path = 'data'

    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)

    for file_path in glob.glob(raw_data_dir_path + '/*.csv'):
        file_name = os.path.basename(file_path)
        file_name = file_name.replace(pathlib.Path(file_name).suffix, "")

        df = pd.read_csv(file_path)
        train_sample, test_sample = train_test_split(df, test_size=0.2)

        model = None

        if args.option == 1:
            model = nb.train(train_sample)
        if args.option == 2:
            model = knn.train(train_sample)

        if model is not None:
            predicted = model.predict(test_sample['conteudo'])
            precision = np.mean(predicted == test_sample['saida'])

            file = open(results_dir_path + "/" + file_name + "_classification.txt", "w")
            file.write("Quantidade de entradas para treino: " + str(len(train_sample.index)) + "\n")
            file.write("Quantidade de entradas para teste: " + str(len(test_sample.index)) + "\n")
            file.write("Precisão: " + str(precision) + "\n")
            file.close()


if __name__ == '__main__':
    main()
