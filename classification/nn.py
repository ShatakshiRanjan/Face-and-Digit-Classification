import random
import util
import math

PRINT = True

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class TwoLayerNetwork:
    """
    A simple two-layer neural network with one hidden layer.
    """
    def __init__(self, legalLabels, max_iterations, hidden_units):
        self.legalLabels = legalLabels
        self.type = "two-layer-network"
        self.max_iterations = max_iterations
        self.hidden_units = hidden_units
        self.weights_hidden = util.Counter()  # Weights from input to hidden layer
        self.weights_output = {}  # Weights from hidden layer to output labels

        # Initialize weights for the hidden layer
        for i in range(hidden_units):
            self.weights_hidden[i] = util.Counter()

        # Initialize weights for the output layer
        for label in self.legalLabels:
            self.weights_output[label] = util.Counter()
            for i in range(hidden_units):
                self.weights_output[label][i] = random.random()

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Training function for the two-layer network.
        """
        self.features = trainingData[0].keys()

        for iteration in range(self.max_iterations):
            if PRINT:
                print("Starting iteration", iteration, "...")
            for i in range(len(trainingData)):
                # Forward pass: compute hidden layer activations
                hidden_activations = util.Counter()
                for h in range(self.hidden_units):
                    hidden_activations[h] = sigmoid(trainingData[i] * self.weights_hidden[h])

                # Compute output layer activations
                activations = util.Counter()
                for label in self.legalLabels:
                    activations[label] = sum(hidden_activations[h] * self.weights_output[label][h] for h in range(self.hidden_units))

                predictedLabel = activations.argMax()
                actualLabel = trainingLabels[i]

                # Update weights if necessary
                if actualLabel != predictedLabel:
                    # Update output layer weights
                    for h in range(self.hidden_units):
                        error_signal = hidden_activations[h] * (1 if actualLabel == label else -1)
                        self.weights_output[actualLabel][h] += error_signal
                        self.weights_output[predictedLabel][h] -= error_signal

                    # Update hidden layer weights
                    for h in range(self.hidden_units):
                        delta = sigmoid_derivative(trainingData[i] * self.weights_hidden[h])
                        for feature, value in trainingData[i].items():
                            self.weights_hidden[h][feature] += delta * value * (1 if actualLabel == label else -1)

    def classify(self, data):
        """
        Classifies each datum using the two-layer network model.
        """
        guesses = []
        for datum in data:
            # Compute hidden layer activations
            hidden_activations = util.Counter()
            for h in range(self.hidden_units):
                hidden_activations[h] = sigmoid(datum * self.weights_hidden[h])

            # Compute output layer activations
            vectors = util.Counter()
            for label in self.legalLabels:
                vectors[label] = sum(hidden_activations[h] * self.weights_output[label][h] for h in range(self.hidden_units))
            guesses.append(vectors.argMax())
        return guesses
    
    def findHighWeightFeaturesHidden(self, hidden_unit):
        """
        Returns a list of the features with the greatest weight connecting to a hidden unit.
        """
        featureWeights = self.weights_hidden[hidden_unit].items()
        topFeatures = sorted(featureWeights, key=lambda feature: feature[1], reverse=True)[:100]
        return [feature[0] for feature in topFeatures]

    def findHighWeightFeaturesOutput(self, label):
        """
        Returns a list of the hidden units with the greatest weight connecting to an output label.
        """
        featureWeights = self.weights_output[label].items()
        topFeatures = sorted(featureWeights, key=lambda feature: feature[1], reverse=True)[:100]
        return [feature[0] for feature in topFeatures]

# Using this network similar to Perceptron
# network = TwoLayerNetwork(legalLabels=[0, 1], max_iterations=10, hidden_units=10)
# network.train(trainingData, trainingLabels, validationData, validationLabels)
