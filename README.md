# Perceptron Classifier with Sparse and Dense Vectors
This repository contains an implementation of a Perceptron for classification of sparse and dense vectors.

# What is a Perceptron?
A Perceptron is a binary classifier that is used in supervised learning. It is a type of a simple neural network that can be used to classify input data into one of two categories. The algorithm learns a linear function that separates the two classes. The learning process involves adjusting the weights of the input variables until the classification error is minimized.

# Implementation Details
The Perceptron in this repository is implemented in Python using the NumPy. It takes in either a dense vector or a sparse vector as input, and uses stochastic gradient descent to learn the weights of the input variables.


The sparse vector are obtained by weighting co-occurrence of two words with Positive Pointwise Mutual Information (PPMI) score, which is a measure of the association between two events (in this application: two words). PMI itself measures the log-likelihod of the joint probabiliy of two words occurring together to the product of their individual probabilities according to the following formula:

PMI $(w_1, w_2) = \log_2(\frac{P(w_1,w_2)}{P(w_1) \cdot P(w_2)})$

In this implementation only positive values of the PMI score are considered and all negative PMI values are set to 0.

The dense vector representations are word embeddings obtained by training Word2Vec with [Gensim](https://radimrehurek.com/gensim/models/word2vec.html).

The model performance is evaluated with 5-fold cross-validation. The accuracy per batch will be printed out with the following command:

        python classifier.py --input-b data/B.txt --input-t data/T.txt --input-text data/input_text.txt

# How to use Perceptron 
- Clone the repository to your local machine
- Install the necessary dependencies by running pip install -r requirements.txt
- Import the Perceptron Classifier from classifier.py and create an instance of the classifier.
- Train the classifier on provided training data (can be found in directory `data`) by calling the `train()` method and passing training data and labels (can also be found in directory `data`: B.txt & T.txt).
- Predict the labels of your test data by calling the `predict()` method and passing in your test data.