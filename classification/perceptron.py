# perceptron.py
# -------------
# Licensing Information: You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

PRINT = True

class PerceptronClassifier:
    """
    Perceptron classifier.
    
    This class implements a simple perceptron classifier that updates weights based
    on classification errors. Each feature of the input data contributes to the decision
    by a weighted sum, and weights are updated based on the perceptron learning rule.
    """
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in self.legalLabels:
            # Initialize weights to zero
            self.weights[label] = util.Counter()

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        Each datum is a util.Counter of features representing a vector of feature values.
        """
        self.features = trainingData[0].keys()  # could be useful later

        for iteration in range(self.max_iterations):
            if PRINT:
                print("Starting iteration", iteration, "...")
            for i in range(len(trainingData)):
                activations = util.Counter()
                for label in self.legalLabels:
                    activations[label] = trainingData[i] * self.weights[label]
                predictedLabel = activations.argMax()
                
                actualLabel = trainingLabels[i]
                if actualLabel != predictedLabel:
                    self.weights[actualLabel] += trainingData[i]
                    self.weights[predictedLabel] -= trainingData[i]

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. Each datum is a util.Counter of features.
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for label in self.legalLabels:
                vectors[label] = self.weights[label] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label.
        """
        featureWeights = self.weights[label].items()
        # Sort features by their weights in descending order and select the top 100
        topFeatures = sorted(featureWeights, key=lambda feature: feature[1], reverse=True)[:100]
        return [feature[0] for feature in topFeatures]  # Returning only the feature names

