"""
This is file is for testing the Naive Bayes classifier
on different Laplace smoothing values to find the best.
"""

import naiveBayes
import util
import random
from util import accuracy_score
import time
import perceptron

# Load data
# Load data
n = 451
n_test = 150
n_val = 301

num_class = 2

height = 70
width = 60

train_data = util.loadDataFile("facedata/facedatatrain", n, width, height)
training_labels = util.loadLabelsFile("facedata/facedatatrainlabels", n)
test_data = util.loadDataFile("facedata/facedatatest", n_test, width, height)
test_labels = util.loadLabelsFile("facedata/facedatatestlabels", n_test)
val_data = util.loadDataFile("facedata/facedatavalidation", n_val, width, height)
validation_labels = util.loadLabelsFile("facedata/facedatavalidationlabels", n_val)

# n = 5000
# n_test = 1000
# n_val = 1000
#
# num_class = 10
#
# height = 28
# width = 28
# #
# train_data = util.loadDataFile("digitdata/trainingimages", n, width, height)
# training_labels = util.loadLabelsFile("digitdata/traininglabels", n)
# test_data = util.loadDataFile("digitdata/testimages", n_test, width, height)
# test_labels = util.loadLabelsFile("digitdata/testlabels", n_test)
# val_data = util.loadDataFile("digitdata/validationimages", n_val, width, height)
# validation_labels = util.loadLabelsFile("digitdata/validationlabels", n_val)

# Initialize classifier
# classifier = naiveBayes.NaiveBayesClassifier(num_class, 2)
#
# # Define Laplace smoothing parameters to tune
# laplace_values = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]
#
# # Tune Laplace smoothing using the validation set
# start = time.time()
# classifier.train_and_tune(train_data, training_labels, val_data, validation_labels, laplace_values)
# end = time.time()

classifier = perceptron.PerceptronClassifier(num_class, height * width)

start = time.time()
classifier.train(train_data, training_labels, 10, 0.01)
end = time.time()

print("Time taken to train and tune:", end - start)

# print("Best Laplace Smoothing Value:", classifier.best_laplace)

# Predict on test data
predictions = classifier.predict(test_data)

# Calculate and display test accuracy
test_accuracy = accuracy_score(test_labels, predictions)
print("Test Accuracy:", test_accuracy)
