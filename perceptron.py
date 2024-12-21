import numpy as np

class PerceptronClassifier:

    def __init__(self, num_classes, num_features):
        self.num_classes = num_classes
        self.num_features = num_features
        self.weights = np.zeros((num_classes, num_features)) # shape: (num_classes, num_features)
        self.biases = np.zeros(num_classes)

    def train(self, X_train, y_train, num_epochs=10, learning_rate=0.01):

        for epoch in range(num_epochs):
            for i in range(len(X_train)):
                x = X_train[i]
                y = y_train[i]

                flattened_x = x.flatten()

                # Compute scores
                scores = np.dot(self.weights, flattened_x) + self.biases

                # Compute predicted class
                predicted_class = np.argmax(scores)

                # Update weights if incorrect
                if predicted_class != y:
                    self.weights[y] += learning_rate * flattened_x
                    self.biases[y] += learning_rate
                    self.weights[predicted_class] -= learning_rate * flattened_x
                    self.biases[predicted_class] -= learning_rate

    def predict(self, X):
        X = np.array(X)  # Ensure input is a NumPy array
        flattened_X = X.reshape(X.shape[0], -1)  # Flatten each test image
        scores = np.dot(flattened_X, self.weights.T) + self.biases
        return np.argmax(scores, axis=1)
