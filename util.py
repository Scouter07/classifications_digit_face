import numpy as np
import os

def loadDataFile(filename, n, width, height):
    """
    Reads `n` data images from a file and returns a NumPy array of shape (n, height, width).
    Truncates if fewer than `n` items are available.
    """
    fin = readlines(filename)
    fin.reverse()
    items = []

    for i in range(n):
        data = []
        for j in range(height):
            if not fin:  # End of file
                print("Truncating at %d examples (maximum)" % i)
                return np.array(items)  # Convert to NumPy array
            data.append(list(fin.pop()))

        # Convert data to integers using the IntegerConversionFunction
        temp = [[IntegerConversionFunction(x) for x in row] for row in data]
        items.append(temp)

    return np.array(items)  # Return NumPy array of shape (n, height, width)

def loadLabelsFile(filename, n):
    """
    Reads `n` labels from a file and returns a NumPy array of integers.
    """
    fin = readlines(filename)
    labels = []

    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        labels.append(int(line))

    return np.array(labels)  # Convert to NumPy array

def readlines(filename):
    """
    Opens a file or reads it from the local directory.
    """
    if os.path.exists(filename):
        return [l.rstrip('\n') for l in open(filename).readlines()]
    else:
        raise FileNotFoundError(f"File {filename} not found.")

def accuracy_score(test_labels, predictions):
    """
    Computes the accuracy score as the fraction of correct predictions.
    """
    test_labels = np.array(test_labels)
    predictions = np.array(predictions)
    return np.mean(test_labels == predictions)

def IntegerConversionFunction(character):
    """
    Converts a character from the file to an integer value:
    - ' ' -> 0
    - '+' -> 2
    - '#' -> 1
    """
    if character == ' ':
        return 0
    elif character == '+':
        return 2  # Edge
    elif character == '#':
        return 1
    else:
        raise ValueError(f"Unexpected character: {character}")

def _test():
    """
    Test function to verify the implementation.
    """
    n = 2
    items = loadDataFile("digitdata/trainingimages", n, 28, 28)
    labels = loadLabelsFile("digitdata/traininglabels", n)
    print("Labels:", labels)
    print("First image data:\n", items[0])
    print("Second image data:\n", items[1])

if __name__ == "__main__":
    _test()
