import random
import util
import math

PRINT = True

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# for readability purposes in a separate function
def derivative_sigmoid(x):
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
        self.weights_inputTohidden = {}  # weights from input to hidden layer
        self.weights_hiddenToOutput = {}  # weights from hidden layer to output labels

        # initializing the weights for the hidden layer
        for i in range(hidden_units):
            self.weights_inputTohidden[i] = util.Counter()

        # initializing the weights for the output layer
        for label in self.legalLabels:
            self.weights_hiddenToOutput[label] = util.Counter()
            for i in range(hidden_units):
                self.weights_hiddenToOutput[label][i] = random.uniform(-0.1, 0.1)  # choose from -0.1 and 0.1 randomly

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Training function for the two-layer network.
        """
        self.features = trainingData[0].keys()
        learning_rate = 0.01 # controlling the process's speed/stability in form of step size

        for iteration in range(self.max_iterations):
            if PRINT:
                print("Starting iteration", iteration, "...")
            for i in range(len(trainingData)):
                # compute using activation func for hidden layer 
                hidden_activations = util.Counter()
                for h in range(self.hidden_units):
                    dot_product = sum(trainingData[i][feature] * self.weights_inputTohidden[h][feature] for feature in self.features)
                    hidden_activations[h] = sigmoid(dot_product)

                # compute using activation fun for output layer
                activations = util.Counter()
                for label in self.legalLabels:
                    dot_product = sum(hidden_activations[h] * self.weights_hiddenToOutput[label][h] for h in range(self.hidden_units))
                    activations[label] = sigmoid(dot_product)

                #ALL OF THIS CODE above was intended for a forward pass
                #dot_product done between feature vector (x) and the weights (w)
                #Backpropagataion will be performed below

                predictedLabel = activations.argMax()
                actualLabel = trainingLabels[i]

                # calculate output errors
                output_errors = {}
                for label in self.legalLabels:
                    output_errors[label] = (1 if actualLabel == label else 0) - activations[label]

                # updating the weights from hidden to output
                # output layer error is based on output error propagated backward 
                for label in self.legalLabels:
                    for h in range(self.hidden_units):
                        self.weights_hiddenToOutput[label][h] += learning_rate * output_errors[label] * hidden_activations[h]


                # determine the errors in the hidden network
                # hidden layer is from derivative of the sigmoid function (to learn complex decision functions/nonlinearities)
                # AND the error propagated from the output layer 
                hidden_errors = {}
                for h in range(self.hidden_units):
                    hidden_errors[h] = sum(output_errors[label] * self.weights_hiddenToOutput[label][h] for label in self.legalLabels) * derivative_sigmoid(hidden_activations[h])

                # Backpropagation = errors form the output back to the hidden 
                # changing the weights from input layer to hidden layer accordingly
                # computes the gradient of the loss function
                for h in range(self.hidden_units):
                    for feature in self.features:
                        self.weights_inputTohidden[h][feature] += learning_rate * hidden_errors[h] * trainingData[i][feature]

    def classify(self, data):
        """
        Classifies each datum using the two-layer network model.
        """
        guesses = []
        for datum in data:
            # determine hidden layer activations
            hidden_activations = util.Counter()
            for h in range(self.hidden_units):
                dot_product = sum(datum[feature] * self.weights_inputTohidden[h][feature] for feature in self.features)
                hidden_activations[h] = sigmoid(dot_product)

            # determine output layer activations
            activations = util.Counter()
            for label in self.legalLabels:
                dot_product = sum(hidden_activations[h] * self.weights_hiddenToOutput[label][h] for h in range(self.hidden_units))
                activations[label] = sigmoid(dot_product)

            guesses.append(activations.argMax())
        return guesses

    def findHighWeightFeaturesHidden(self, hidden_unit):
        """
        Returns a list of the features with the greatest weight connecting to a hidden unit.
        """
        featureWeights = self.weights_inputTohidden[hidden_unit].items()
        topFeatures = sorted(featureWeights, key=lambda feature: feature[1], reverse=True)[:100]
        return [feature[0] for feature in topFeatures]

    def findHighWeightFeaturesOutput(self, label):
        """
        Returns a list of the hidden units with the greatest weight connecting to an output label.
        """
        featureWeights = self.weights_hiddenToOutput[label].items()
        topFeatures = sorted(featureWeights, key=lambda feature: feature[1], reverse=True)[:100]
        return [feature[0] for feature in topFeatures]
