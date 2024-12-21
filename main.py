from numpy.random import laplace

import naiveBayes
import perceptron
import util
import random
import time
import matplotlib.pyplot as plt
import numpy as np


def data_partition(data, labels, k):
    n = len(data)
    # Randomly select k unique indices
    choices = np.random.choice(n, k, replace=False)
    # Use NumPy indexing to select the subset
    new_data = data[choices]
    new_labels = labels[choices]
    return new_data, new_labels


def analysis(classifier, training_data, training_labels, test_data, test_labels, validation_data, validation_labels, epochs=10, clas="", dat=""):
    n = len(training_data)
    data_points = [int((x / 10) * n) for x in range(1, 11)]
    avg_accuracies = []
    avg_times = []
    standard_deviations = []
    errors = []

    for i in data_points:
        accuracies = []
        times = []
        error = []

        for j in range(epochs):
            new_training_data, new_training_labels = data_partition(training_data, training_labels, i)

            start = 0
            end = 0

            if clas == "naiveBayes":
                laplace_smoothing = [0.001]
                start = time.time()
                classifier.train_and_tune(new_training_data, new_training_labels, validation_data, validation_labels, laplace_smoothing)
                end = time.time()
            elif clas == "perceptron":
                learning_rate = 0.01
                start = time.time()
                classifier.train(new_training_data, new_training_labels, epochs, learning_rate)
                end = time.time()

            y_pred = classifier.predict(test_data)
            accuracy = util.accuracy_score(test_labels, y_pred)

            print(f"Data Points: {i}, Accuracy: {accuracy}, Time: {end - start}, Epoch: {j}, Error: {1 - accuracy}")

            accuracies.append(accuracy)
            times.append(end - start)
            error.append(1 - accuracy)

        avg_accuracies.append(sum(accuracies) / len(accuracies))
        avg_times.append(sum(times) / len(times))
        standard_deviations.append(np.std(accuracies))
        errors.append(sum(error) / len(error))

    plt.plot(data_points, avg_accuracies)
    plt.xlabel("Number of Training Data Points")
    plt.ylabel("Average Accuracy")
    plt.title(f"{dat} {clas} Classifier   Average Accuracy vs. Number of Training Data Points")

    # Save graphs to results folder
    plt.savefig(f"results/{clas}_{dat}_accuracy.png")
    plt.clf()

    plt.plot(data_points, avg_times)
    plt.xlabel("Number of Training Data Points")
    plt.ylabel("Average Time")
    plt.title(f"{dat} {clas} Classifier   Average Time vs. Number of Training Data Points")
    plt.savefig(f"results/{clas}_{dat}_time.png")
    plt.clf()

    plt.plot(data_points, standard_deviations)
    plt.xlabel("Number of Training Data Points")
    plt.ylabel("Standard Deviation")
    plt.title(f"{dat} {clas} Classifier   Standard Deviation vs. Number of Training Data Points")
    plt.savefig(f"results/{clas}_{dat}_std.png")
    plt.clf()

    plt.plot(data_points, errors)
    plt.xlabel("Number of Training Data Points")
    plt.ylabel("Error")
    plt.title(f"{dat} {clas} Classifier   Error vs. Number of Training Data Points")
    plt.savefig(f"results/{clas}_{dat}_error.png")
    plt.clf()

    return




def main():
    x = int(input("Choose Dataset\n"
                  "1 - Digit Classifier\n"
                  "2 - Face Classifier\n"))

    num_class = -1
    edge_weight = 2
    epochs = 10
    clas = ""
    dat = ""
    if x == 1:
        # Load data
        n = 5000
        n_test = 1000
        n_val = 1000

        num_class = 10

        height = 28
        width = 28

        training_data = util.loadDataFile("digitdata/trainingimages", n, width, height)
        training_labels = util.loadLabelsFile("digitdata/traininglabels", n)
        test_data = util.loadDataFile("digitdata/testimages", n_test, width, height)
        test_labels = util.loadLabelsFile("digitdata/testlabels", n_test)
        validation_data = util.loadDataFile("digitdata/validationimages", n_val, width, height)
        validation_labels = util.loadLabelsFile("digitdata/validationlabels", n_val)

        dat = "digit"
    elif x == 2:
        # Load data
        n = 451
        n_test = 150
        n_val = 301

        num_class = 2

        height = 70
        width = 60

        training_data = util.loadDataFile("facedata/facedatatrain", n, width, height)
        training_labels = util.loadLabelsFile("facedata/facedatatrainlabels", n)
        test_data = util.loadDataFile("facedata/facedatatest", n_test, width, height)
        test_labels = util.loadLabelsFile("facedata/facedatatestlabels", n_test)
        validation_data = util.loadDataFile("facedata/facedatavalidation", n_val, width, height)
        validation_labels = util.loadLabelsFile("facedata/facedatavalidationlabels", n_val)

        dat = "face"
    else:
        print("Invalid dataset choice")
        return

    y = int(input("Choose Classifier\n"
                  "1 - Naive Bayes Classifier\n"
                  "2 - Perceptron Classifier\n"))

    classifier = None
    if y == 1:
        classifier = naiveBayes.NaiveBayesClassifier(num_class, edge_weight)
        clas = "naiveBayes"
    elif y == 2:
        classifier = perceptron.PerceptronClassifier(num_class, height * width)
        clas = "perceptron"
    else:
        print("Invalid classifier choice")
        return



    analysis(classifier, training_data, training_labels, test_data, test_labels, validation_data, validation_labels, epochs, clas, dat)


if __name__ == "__main__":
    main()