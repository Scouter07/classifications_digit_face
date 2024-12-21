import numpy as np
from util import accuracy_score

class NaiveBayesClassifier:
    def __init__(self, num_classes, edge_weight=2):
        self.num_classes = num_classes
        self.class_priors = None
        self.log_pixel_likelihoods = None
        self.edge_weight = edge_weight
        self.best_laplace = None
        self.pixel_counts = None
        self.class_pixel_totals = None
        self.height = None
        self.width = None

    def train_and_tune(self, X_train, y_train, X_val, y_val, laplace_values):

        # Compute class counts and priors
        class_counts = np.bincount(y_train, minlength=self.num_classes)
        num_images = len(y_train)
        self.class_priors = np.log(class_counts / num_images)

        # Store image dimensions
        self.height, self.width = X_train.shape[1], X_train.shape[2]

        # Precompute pixel counts for all classes once
        # pixel_counts shape: (num_classes, height, width, 3)
        self.pixel_counts = np.zeros((self.num_classes, self.height, self.width, 3), dtype=np.float64)

        for c in range(self.num_classes):
            class_mask = (y_train == c)
            class_images = X_train[class_mask]  # shape: (#images_of_class, height, width)

            # Count occurrences of each pixel value (0,1,2) for this class
            # Value 0 counts
            val0_counts = np.sum(class_images == 0, axis=0)
            # Value 1 counts
            val1_counts = np.sum(class_images == 1, axis=0)
            # Value 2 counts (edge), weighted
            val2_counts = np.sum(class_images == 2, axis=0) * self.edge_weight

            # Combine into pixel_counts for this class
            self.pixel_counts[c, :, :, 0] = val0_counts
            self.pixel_counts[c, :, :, 1] = val1_counts
            self.pixel_counts[c, :, :, 2] = val2_counts

        # Compute total pixel counts per class
        # Each image has height*width pixels, but edges are weighted
        self.class_pixel_totals = np.sum(self.pixel_counts, axis=(1,2,3))

        best_accuracy = -1
        best_log_pixel_likelihoods = None

        # Try each Laplace smoothing value
        for laplace_smoothing in laplace_values:
            # Compute log-likelihoods given this smoothing
            log_pixel_likelihoods = self.compute_log_likelihoods(laplace_smoothing)

            # Validate
            y_pred = self.predict(X_val, log_pixel_likelihoods)
            accuracy = accuracy_score(y_val, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_log_pixel_likelihoods = log_pixel_likelihoods
                self.best_laplace = laplace_smoothing

        # Store the best likelihoods
        self.log_pixel_likelihoods = best_log_pixel_likelihoods

    def compute_log_likelihoods(self, laplace_smoothing):
        # Apply Laplace smoothing:
        # log P(pixel = v | class = c) = log((count + laplace_smoothing) / (total + 3*laplace_smoothing))
        denominator = self.class_pixel_totals[:, None, None, None] + 3 * laplace_smoothing
        numerator = self.pixel_counts + laplace_smoothing
        return np.log(numerator / denominator)

    def predict(self, X_test, log_pixel_likelihoods=None):
        if log_pixel_likelihoods is None:
            log_pixel_likelihoods = self.log_pixel_likelihoods

        X_test = np.array(X_test)  # Ensure numpy array
        n_samples = X_test.shape[0]
        predictions = np.zeros(n_samples, dtype=int)

        # Precompute index arrays for vectorized indexing
        row_indices = np.arange(self.height)[:, None]
        col_indices = np.arange(self.width)

        for idx in range(n_samples):
            img = X_test[idx]

            # log_posteriors[c] = logP(C) + sum of log pixel likelihoods
            # Vectorized by numpy is too good. I love numpy.
            log_posteriors = self.class_priors.copy()

            # Extract pixel likelihoods for each class in a vectorized manner
            # shape: (num_classes, height, width)
            # We'll use advanced indexing over the last dimension with `img`.
            for c in range(self.num_classes):
                log_pixel_vals = log_pixel_likelihoods[c, row_indices, col_indices, img]
                log_posteriors[c] += np.sum(log_pixel_vals)

            predictions[idx] = np.argmax(log_posteriors)

        return predictions
