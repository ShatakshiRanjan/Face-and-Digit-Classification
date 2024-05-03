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
        self.weights_hidden = {}  # Weights from input to hidden layer
        self.weights_output = {}  # Weights from hidden layer to output labels

        # Initialize weights for the hidden layer
        for i in range(hidden_units):
            self.weights_hidden[i] = util.Counter()

        # Initialize weights for the output layer
        for label in self.legalLabels:
            self.weights_output[label] = util.Counter()
            for i in range(hidden_units):
                self.weights_output[label][i] = random.uniform(-0.1, 0.1)  # Small random values

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Training function for the two-layer network.
        """
        self.features = trainingData[0].keys()
        learning_rate = 0.01

        for iteration in range(self.max_iterations):
            if PRINT:
                print("Starting iteration", iteration, "...")
            for i in range(len(trainingData)):
                # Forward pass: compute hidden layer activations
                hidden_activations = util.Counter()
                for h in range(self.hidden_units):
                    dot_product = sum(trainingData[i][feature] * self.weights_hidden[h][feature] for feature in self.features)
                    hidden_activations[h] = sigmoid(dot_product)

                # Compute output layer activations
                activations = util.Counter()
                for label in self.legalLabels:
                    dot_product = sum(hidden_activations[h] * self.weights_output[label][h] for h in range(self.hidden_units))
                    activations[label] = sigmoid(dot_product)

                predictedLabel = activations.argMax()
                actualLabel = trainingLabels[i]

                # Calculate output errors
                output_errors = {}
                for label in self.legalLabels:
                    output_errors[label] = (1 if actualLabel == label else 0) - activations[label]

                # Update weights from hidden to output
                for label in self.legalLabels:
                    for h in range(self.hidden_units):
                        self.weights_output[label][h] += learning_rate * output_errors[label] * hidden_activations[h]

                # Calculate hidden errors
                hidden_errors = {}
                for h in range(self.hidden_units):
                    hidden_errors[h] = sum(output_errors[label] * self.weights_output[label][h] for label in self.legalLabels) * sigmoid_derivative(hidden_activations[h])

                # Update weights from input to hidden
                for h in range(self.hidden_units):
                    for feature in self.features:
                        self.weights_hidden[h][feature] += learning_rate * hidden_errors[h] * trainingData[i][feature]

    def classify(self, data):
        """
        Classifies each datum using the two-layer network model.
        """
        guesses = []
        for datum in data:
            # Compute hidden layer activations
            hidden_activations = util.Counter()
            for h in range(self.hidden_units):
                dot_product = sum(datum[feature] * self.weights_hidden[h][feature] for feature in self.features)
                hidden_activations[h] = sigmoid(dot_product)

            # Compute output layer activations
            activations = util.Counter()
            for label in self.legalLabels:
                dot_product = sum(hidden_activations[h] * self.weights_output[label][h] for h in range(self.hidden_units))
                activations[label] = sigmoid(dot_product)

            guesses.append(activations.argMax())
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

# Example usage:
# network = TwoLayerNetwork(legalLabels=[0, 1], max_iterations=10, hidden_units=10)
# network.train(trainingData, trainingLabels, validationData, validationLabels, learning_rate=0.01)
