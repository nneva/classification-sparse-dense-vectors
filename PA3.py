#/usr/bin/python3
# pa3.py
# author: Nevena Nikolic

import argparse
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Coroutine
import math
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from typing import List, Set, DefaultDict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-b", type=str, 
                        help="Input file with B-words.", required=True)
    parser.add_argument("--input-t", type=str, 
                        help="Input file with T-words.", required=True)
    parser.add_argument("--input-text", type=str, 
                        help="Input file with text.", required=True)

    args = parser.parse_args()

    return args



class SparseVector(object):
    pass



class DenseVector(object):
    pass


class Perceptron(object):

    def __init__(self, data: tuple) -> None:
        self.data = data


    def _get_data(input_file) -> tuple:

        file = pd.read_csv(input_file, sep="\t")
        co_matrix =  file.to_numpy()

        return (file.columns.array, co_matrix, file.Label.values, file.Word.values)


    def sigmoid(product, x) -> float:

        return 1 / (1 + np.exp(-product * x))


    def compute_weights(self) -> list:

        weights = [0.0] * (len(self.data[1][1]) - 2)
        bias_unit = np.ones((len(self.data[1][:, 1])))
        initial_features = np.delete(self.data[1], (0, 84, 85) , axis=1)
        input_features = np.c_[initial_features, bias_unit]
        # "WAR" is mapped to 1, "PEACE" is mapped to 0.
        desired_output = list(map(lambda x: 1 if x == 'WAR' else 0, self.data[2]))

        # Stopping criterion determined on manual run.
        for it in range(100):
            if it % 10 == 5:
                print("results after", str(it + 1), "iterations:")

            for idx, input_feature in enumerate(input_features):
                product = (np.dot(input_feature, weights))
                sigmoid = Perceptron.sigmoid(product, 2)
                if it % 10 == 5:
                    print("true result:", desired_output[idx], "output:", 1 if sigmoid > 0.5 else 0)
                
                error = desired_output[idx] - sigmoid
                if abs(error) > 0.0:
                    for i, val in enumerate(input_feature):
                        # 0.2 is the learning rate which we can choose freely between 0 and 1.
                        weights[i] += val * error * 0.2
            
        return weights


def main():

    data = Perceptron._get_data(args.input_file)
    perceptron = Perceptron(data)
    weights = perceptron.compute_weights()
    pass

if __name__ == "__main__":
    main()