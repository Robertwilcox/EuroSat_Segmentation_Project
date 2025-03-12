"""
test_evaluation.py
------------------
Unit test for evaluating the confusion matrix computation.

Author: Robert Wilcox
Date: 03.10.25

This module contains a test that computes a confusion matrix using scikit-learn
and verifies it against an expected result using numpy's testing utilities.
"""

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix


def test_confusion_matrix():
    """
    Test that the confusion matrix is computed correctly.
    """
    # True class labels for a sample set of predictions.
    y_true = np.array(["Forest", "Highway", "Forest", "Residential", "SeaLake", "Forest"])
    # Predicted class labels for the same set.
    y_pred = np.array(["Forest", "Forest", "Forest", "Residential", "SeaLake", "Highway"])
    
    # Define the order of class labels.
    labels = ["Forest", "Highway", "Residential", "SeaLake"]
    # Compute the confusion matrix using the specified labels.
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Print the computed confusion matrix for debugging purposes.
    print("Computed Confusion Matrix:")
    print(cm)
    
    # Define the expected confusion matrix.
    expected_cm = np.array([
        [2, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    # Assert that the computed confusion matrix matches the expected one.
    np.testing.assert_array_equal(cm, expected_cm)


if __name__ == "__main__":
    pytest.main()
