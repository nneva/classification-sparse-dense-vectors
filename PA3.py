#/usr/bin/python3
# pa3.py
# author: Nevena Nikolic

import argparse
from argparse import ArgumentParser
import numpy as np
from numpy.typing import ArrayLike
import re
import math
import gensim
from gensim.models import KeyedVectors, Word2Vec
from typing import List, Set, DefaultDict, TextIO, Tuple
from collections import defaultdict
from collections.abc import Coroutine

"""
Apologies for not submitting my script with more comments, as promised. 
Working on correcting the error with dimensions, and adding comments and docs.
Anyways, feel free to add/ask any additional remarks/questions as soon as I finish. 

Nice BLEU implementation, btw. :)
"""

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-b", type=str, 
                        help="Path to file with B-words.", required=True)
    parser.add_argument("--input-t", type=str, 
                        help="Path to file with T-words.", required=True)
    parser.add_argument("--input-text", type=str, 
                        help="Path to file with text.", required=True)

    args = parser.parse_args()

    return args


def preprocess_raw(raw_file: TextIO) -> str:
    """
    Get clean text for further processing.
    :param raw_file: Input file.
    :return: Text as string. 
    """
    # define separators for splitting the text
    sep = ["\n\n\n\nBOOK ONE: 1805\n\n", "End of the Project Gutenberg EBook of War and Peace, by Leo Tolstoy"]

    part_1 = raw_file.read().split(sep[0])
    part_2 = part_1[1].split(sep[1])
    main_part = re.sub(r"CHAPTER\s[X | I | V]" , " ",  part_2[0]).lstrip().lower()

    final_part = ''.join(re.split(r"[\.\",':()-?/\t\n]", main_part)).lower()

    return final_part


class DenseVectors:
    """Class for creating dense vector representations."""

    def preprocess_b(self, b_words: TextIO) -> List[str]:
        """
        Preprocess file with context words.
        :param b_file: Text file containing context words.
        :return: Context words as list of strings.
        """
        with open(b_words, "r") as b:
            basis_words = [line.strip() for line in b]

            return basis_words
    
    def preprocess_t(self, t_words: TextIO) -> List[str]:
        """
        Preprocess file with target words.
        :param t_file: Text file containing target words.
        :return: Target words as list of strings.
        """
        with open(t_words, "r") as t:
            trg_words = [line.split("\t")[0] for line in t]
        
            return trg_words


    def train(self, corpus_file: str, t_words: TextIO) -> ArrayLike:
        """
        Train word embeddings.
        :param corpus_file: One of five batches of the input text.
        :param b_file: Text file containing context words.
        :return: Trained word embeddings (vectors) as matrix. 
        """
        # initialize a model with args: vector size, skip-gram, number of processes, window size, min. frequency of the word
        model = Word2Vec(vector_size=84, sg=1, workers=1, window=2, min_count=1)
        model.build_vocab(corpus_file=corpus_file)
        model.train(corpus_file=corpus_file, epochs=60, total_words=model.corpus_count)
        # store words as keys and vectors as their respective values
        word_vectors = model.wv
        word_vectors.save("word2vec.wordvectors")
        # store vector size for matrix initialization
        size = model.vector_size
        del model
        wv = KeyedVectors.load("word2vec.wordvectors", mmap="r")
        # initialize zero-matrix of size 45 x 84
        dense_matrix = np.zeros(shape=(45, size))
        # for every target word update matrix with target word's respective vector 
        for idx_t, t_word in enumerate(self.preprocess_t(t_words)):
            vector = wv[t_word]
            dense_matrix[idx_t] = vector

        return dense_matrix


class SparseVectors(object):
    """Class for creating sparse vector representations."""

    def get_counts(trg_words: List[str],
                    basis_words: List[str],
                    line: str,
                    bigrams: DefaultDict,
                    trg_unigrams: DefaultDict,
                    basis_unigrams: DefaultDict
                    ) -> Coroutine:
        """
        Store frequency counts of target and context words, and co-occurrence counts of a target and context word.
        :param trg_words: List of target words.
        :param basis_words: List of context words.
        :param line: Line of the input text as string.
        :param bigrams: Dictionary to store co-occurrence counts of a target and context word.
        :param trg_unigrams: Dictionary to store frequency counts of target words.
        :param basis_unigrams: Dictionary to store frequency counts of context words.
        """
        while True:
            line = (yield)
            for trg_word in trg_words:
                if trg_word in line:
                    trg_unigrams[trg_word] += 1
                    for basis_word in basis_words:
                        if basis_word in line: 
                            basis_unigrams[basis_word] += 1
                            if trg_word +' '+ basis_word in line or basis_word +' '+ trg_word in line:
                                bigrams[trg_word +' '+ basis_word] += 1


    def compute_PPMI(self, trg_words: List[str],
                    basis_words: List[str],
                    bigrams: DefaultDict,
                    trg_unigrams: DefaultDict,
                    basis_unigrams: DefaultDict
                    ) -> ArrayLike:
        """
        Compute PPMI based on probabilities of target and context words.
        :param trg_words: List of target words.
        :param basis_words: List of context words.
        :param bigrams: Dictionary with co-occurrence counts of a target and context word.
        :param trg_unigrams: Dictionary with frequency counts of target words.
        :param basis_unigrams: Dictionary with frequency counts of context words.
        :return: Matrix with PPMI values. 
        """
        
        PPMI_matrix = np.zeros(shape=(len(trg_words), len(basis_words)))

        for idx_trg, trg_word in enumerate(trg_words):
            for idx_basis, basis_word in enumerate(basis_words):
                count = bigrams[trg_word + ' ' + basis_word]
                prob_bigram = count / sum(bigrams.values())
                prob_trg_word = trg_unigrams[trg_word] / sum(trg_unigrams.values())
                prob_basis_word = basis_unigrams[basis_word] / sum(basis_unigrams.values())
                PPMI = max(math.log2(prob_bigram / (prob_trg_word * prob_basis_word)), 0) if trg_unigrams[trg_word] \
                    and basis_unigrams[basis_word] and prob_bigram != 0 else 0
                PPMI_matrix[idx_trg][idx_basis] = PPMI

        return PPMI_matrix


    def get_cooccurrence_matrix(trg_words: List[str],
                                basis_words: List[str], 
                                line: str,
                                CO_matrix: ArrayLike
                                ) -> Coroutine:
        while True:
            line = (yield)
            for idx_trg, trg_word in enumerate(trg_words):
                if trg_word in line:
                    sent = line.split()
                    for word in sent:
                        if word == trg_word:
                            idx = sent.index(word)
                            #window size equal to 5
                            window = [sent[idx - 2], sent[idx - 1], sent[idx], sent[idx + 1], sent[idx + 2]] \
                                if idx < len(sent) - 2 and idx >= 2 else ''
                            for word in window:
                                if word in set(basis_words):
                                    idx_basis = basis_words.index(word)
                                    CO_matrix[idx_trg][idx_basis] += 1


    def get_weighted_CO_matrix(self, raw_text: str) -> ArrayLike:
        
        bigrams = defaultdict(int)
        trg_unigrams = defaultdict(int)
        basis_unigrams = defaultdict(int)


        with open("data/pa3_B.txt", "r") as b_words, open("data/pa3_T.txt", "r") as t_words:
            basis_words = [line.strip() for line in b_words.readlines()]
            trg_words = [line.split("\t")[0] for line in t_words.readlines()]  

        CO_matrix = np.ones(shape=(len(trg_words), len(basis_words)))
        
        for line in raw_text.splitlines():
            counts = SparseVectors.get_counts(trg_words, basis_words, line, bigrams, trg_unigrams, basis_unigrams)
            # get element from the iterator
            next(counts)
            # send element to coroutine
            counts.send(line)
            co_occurrence = SparseVectors.get_cooccurrence_matrix(trg_words, basis_words, line, CO_matrix)
            next(co_occurrence)
            co_occurrence.send(line)

        sparse_vector = SparseVectors()
        PPMI_matrix = sparse_vector.compute_PPMI(trg_words, basis_words, bigrams, trg_unigrams, basis_unigrams)

        weighted_CO_matrix = CO_matrix * PPMI_matrix

        return weighted_CO_matrix


class Perceptron:

    def __init__(self, threshold = 0.5, learning_rate = 0.2):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.weights = []
    

    def sigmoid(x) -> float:

        return 1 / (1 + np.exp(-2 * x))

        
    def train(self, inputs, t_words):
        iterations = 0

        bias_unit = np.ones((len(inputs[:, 1])), dtype="int8")
        input_features = np.c_[inputs, bias_unit]
        # "WAR" is mapped to 1, "PEACE" is mapped to 0.
        labels = [word.split("\t")[1].strip() for word in open(t_words).readlines()]
        desired_output = list(map(lambda x: 1 if x == 'WAR' else 0, labels))
            
        while True:
            iterations += 1

            for idx, input_feature in enumerate(input_features):
                # writing interim results into the file
                with open("input_features.txt", "a") as out:
                    out.write(str(input_feature) + "\n")
                weights =  [0.0] * len(input_feature) 
                result = Perceptron.sigmoid((np.dot(input_feature, weights)))
                error = desired_output[idx] - result

                if abs(error) > 0.1:

                    for idx, value in enumerate(input_feature):
                        weights[idx] += self.learning_rate * error * value

                    self.weights.append(weights)

            if iterations == 45:
                break

    # check ndim!
    def predict(self, inputs):
        predicted = []
        bias_unit = np.ones((len(inputs[:, 1])), dtype="int8")
        input_features = np.c_[inputs, bias_unit]

        for input_feature, weight in zip(input_features, self.weights):
            # remap to "WAR" and "PEACE", needed for computing the accuracy
            p = "WAR" if np.dot(input_feature, weight) > self.threshold else "PEACE"
            predicted.append(p)

        return predicted


    def evaluate_cross_val(self, t_words, b_words, input_text):
        raw_text = preprocess_raw(open(input_text))
        batch_size = len(raw_text) // 5
        batches = []

        sparse_vectors = SparseVectors()
        dense_vectors = DenseVectors()
        perceptron = Perceptron()

        labels = [word.split("\t")[1].strip() for word in open(t_words).readlines()]
        total_sparse = 0
        total_dense = 0

        for idx in range(5):
            # split initial data set into batches for cross-validation
            batch = raw_text[idx * batch_size : (idx + 1) * batch_size]
            rest = raw_text[0: idx * batch_size] + raw_text[(idx + 1) * batch_size :]
            
            # compute accuracy for sparse vectors
            weighted_CO_matrix = sparse_vectors.get_weighted_CO_matrix(rest)
            weighted_CO_matrix_test = sparse_vectors.get_weighted_CO_matrix(batch)
            perceptron.train(inputs=weighted_CO_matrix, t_words=t_words)
            predict_sparse = perceptron.predict(inputs=weighted_CO_matrix_test)

            correct = 0
            for predicted, label in zip(predict_sparse, labels):
                if predicted == label:
                    correct += 1

            accuracy = correct / len(labels)
            total_sparse += accuracy
            print(f"Accuracy of batch {idx + 1} is {accuracy} for sparse vectors.")

            # compute accuracy for dense vectors
            with open("batch.txt", "w") as out_batch, open("rest.txt", "w") as out_rest:
                out_batch.write(batch)
                out_rest.write(rest)

            train_dense_vectors = dense_vectors.train("rest.txt", b_words)
            train_dense_vectors_test = dense_vectors.train("batch.txt", b_words)
            train_dense = perceptron.train(inputs=train_dense_vectors, t_words=t_words)
            predict_dense = perceptron.predict(inputs=train_dense_vectors_test)

            correct_dense = 0
            for predicted, label in zip(predict_dense, labels):
                if predicted == label:
                    correct_dense += 1

            accuracy = correct_dense / len(labels)
            total_dense += accuracy
            print(f"Accuracy of batch {idx + 1} is {accuracy} for dense vectors.")

            
        print(f"Average accuracy is {total_sparse / 5} for sparse vectors.")
        print(f"Average accuracy is {total_dense / 5} for dense vectors.")    


def evaluate_single(t_words, b_words, input_text):
    labels = [word.split("\t")[1].strip() for word in open(t_words).readlines()]

    perceptron = Perceptron()

    sparse_vectors = SparseVectors()
    raw_text = preprocess_raw(open(input_text))
    weighted_CO_matrix = sparse_vectors.get_weighted_CO_matrix(raw_text) 
    train_sparse = perceptron.train(inputs=weighted_CO_matrix, t_words=t_words)
    predict_sparse = perceptron.predict(inputs=weighted_CO_matrix)

    correct_single_sparse = 0

    for predicted, label in zip(predict_sparse, labels):
        if predicted == label:
            correct_single_sparse += 1

    accuracy_single_sparce = correct_single_sparse / len(labels)

    dense_vectors = DenseVectors()
    train_dense_vectors = dense_vectors.train(input_text, b_words, t_words)
    train_dense = perceptron.train(inputs=train_dense_vectors, t_words=t_words)
    predict_dense = perceptron.predict(inputs=train_dense_vectors)

    correct_single_dense = 0

    for predicted, label in zip(predict_dense, labels):
        if predicted == label:
            correct_single_dense += 1

    accuracy_single_dense = correct_single_dense / len(labels)

    print(f"Single accuracy is {accuracy_single_sparce} for sparse vectors.")
    print(f"Single accuracy is {accuracy_single_dense} for dense vectors.")


def main():
    args = parse_args()
    b_words = args.input_b
    t_words = args.input_t
    input_text = args.input_text

    result_single = evaluate_single(t_words, b_words, input_text)
    perceptron = Perceptron()
    perceptron.evaluate_cross_val(t_words, b_words, input_text)


if __name__ == "__main__":
    main()