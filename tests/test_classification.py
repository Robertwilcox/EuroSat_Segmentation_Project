"""
test_classification.py
----------------------
Unit tests for image feature extraction and land use classification.

Author: Robert Wilcox
Date: 03.10.25

This module contains tests using pytest to validate the functionality of the
feature extraction functions and the LandUseClassifier from the classification module.
It tests that:
 - Color feature extraction returns a 6-length vector.
 - The classifier can be trained and its predictions are valid based on a sample CSV.
"""

import os
import cv2
import numpy as np
import pytest
import pandas as pd

from src.classification.classifier import extract_color_features, LandUseClassifier


def load_image_and_extract_features(img_path):
    """
    Load an image from a given path, convert it to RGB, and extract color features.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        numpy.ndarray or None: Extracted color features if image is loaded,
                               otherwise None.
    """
    # Read image using OpenCV.
    img = cv2.imread(img_path)
    if img is None:
        return None
    # Convert image from BGR to RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Extract and return color features.
    return extract_color_features(img)


def test_feature_extraction():
    """
    Test that color feature extraction returns a valid 6-length vector.
    """
    img_path = os.path.join(
        os.getcwd(), "data", "raw", "EuroSAT", "Forest", "Forest_1.jpg"
    )
    features = load_image_and_extract_features(img_path)
    # Ensure features were extracted.
    assert features is not None, "Feature extraction failed."
    # Check that the feature vector length is exactly 6.
    assert features.shape[0] == 6, "Extracted feature vector length is not 6."


def test_classifier_training_and_prediction():
    """
    Test classifier training and prediction using a sample of the dataset.
    """
    csv_path = os.path.join(os.getcwd(), "data", "raw", "EuroSAT", "train.csv")
    df = pd.read_csv(csv_path, index_col=0)
    # Sample 30 rows for testing.
    sample_df = df.sample(n=30, random_state=42)

    X = []
    y = []
    # Loop over each sampled row to extract features and labels.
    for idx, row in sample_df.iterrows():
        img_path = os.path.join(os.getcwd(), "data", "raw", "EuroSAT", row["Filename"])
        features = load_image_and_extract_features(img_path)
        if features is not None:
            X.append(features)
            y.append(row["ClassName"])

    X = np.array(X)
    y = np.array(y)
    # Ensure that some features have been extracted.
    assert X.shape[0] > 0, "No features extracted."

    from sklearn.model_selection import train_test_split
    # Split the dataset into training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Initialize and train the classifier.
    classifier = LandUseClassifier()
    classifier.train(X_train, y_train)

    # Get predictions for the validation set.
    preds = classifier.predict(X_val)
    valid_classes = set(sample_df["ClassName"].unique())
    # Check that each prediction is among the valid classes.
    for p in preds:
        assert p in valid_classes, f"Prediction {p} is not a valid class."


if __name__ == "__main__":
    pytest.main()
