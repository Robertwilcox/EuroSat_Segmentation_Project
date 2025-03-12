"""
classifier.py
-------------
A module for extracting features from images and classifying land use.

Author: Robert Wilcox
Date: 03.10.25

This script contains functions to extract color, texture, and segmented features from
images, and defines two classifiers: LandUseClassifier and RobustLandUseClassifier.
The classifiers are built using scikit-learn's SVC and Pipeline, and include options
for hyperparameter tuning.
"""

import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import local_binary_pattern
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Import segmentation functions from the segmentation module.
from src.segmentation.kmeans import kmeans_segmentation
from src.segmentation.fuzzy_cmeans import fuzzy_cmeans_segmentation


def extract_color_features(image):
    """
    Extract basic color features from an image.
    
    Computes the mean and standard deviation for each RGB channel.

    Parameters:
        image (numpy.ndarray): Image of shape (H, W, 3) in RGB format.

    Returns:
        numpy.ndarray: 1D array containing the means and standard deviations (length 6).
    """
    # Calculate mean for each color channel.
    means = np.mean(image, axis=(0, 1))
    # Calculate standard deviation for each color channel.
    stds = np.std(image, axis=(0, 1))
    # Concatenate means and standard deviations into a single feature vector.
    return np.concatenate([means, stds])


def extract_segmented_features(image, k=2, m=2, method='kmeans'):
    """
    Segment the image and extract aggregated features from each segment.

    Parameters:
        image (numpy.ndarray): Image of shape (H, W, 3) in RGB format.
        k (int): Number of segments/clusters.
        m (int): Fuzziness parameter (used only for fuzzy segmentation).
        method (str): 'kmeans' or 'fuzzy' segmentation.

    Returns:
        numpy.ndarray: A 1D array representing aggregated features from the segments.
    """
    # Perform segmentation based on the selected method.
    if method == 'kmeans':
        labels, centroids, _ = kmeans_segmentation(image, k)
    elif method == 'fuzzy':
        labels, centroids, _ = fuzzy_cmeans_segmentation(image, k, m)
    else:
        raise ValueError("Invalid segmentation method. Choose 'kmeans' or 'fuzzy'.")

    # Initialize list to store features for each segment.
    features_list = []
    for i in range(k):
        # Create a mask for the current segment.
        mask = (labels == i)
        # Skip if no pixels are assigned to this segment.
        if np.sum(mask) == 0:
            continue
        # Create a copy of the image and zero out pixels not in the current segment.
        cluster_img = image.copy()
        cluster_img[~mask] = 0
        # Extract combined features (color and texture) for the segmented region.
        features = extract_combined_features(cluster_img)
        features_list.append(features)

    if features_list:
        # Aggregate features by computing the mean across segments.
        aggregated_features = np.mean(np.array(features_list), axis=0)
        return aggregated_features
    else:
        return None


def extract_texture_features(image, P=8, R=1.0):
    """
    Extract texture features using Local Binary Patterns (LBP).

    Converts the image to grayscale, computes the LBP, and returns a normalized histogram.

    Parameters:
        image (numpy.ndarray): Image of shape (H, W, 3) in RGB format.
        P (int): Number of circularly symmetric neighbor set points.
        R (float): Radius of circle.

    Returns:
        numpy.ndarray: 1D array representing the normalized LBP histogram with a fixed length.
    """
    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Compute the LBP image using uniform method.
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    # Define number of bins for the LBP histogram.
    n_bins = 10
    # Compute a normalized histogram of LBP values.
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_combined_features(image):
    """
    Combine color and texture features into a single feature vector.

    Parameters:
        image (numpy.ndarray): Image of shape (H, W, 3) in RGB format.

    Returns:
        numpy.ndarray: 1D array containing both color and texture features.
    """
    # Extract color features (mean and standard deviation).
    color_features = extract_color_features(image)  # length 6
    # Extract texture features using LBP.
    texture_features = extract_texture_features(image)  # typically length 10 for uniform LBP with P=8
    # Concatenate both feature sets.
    combined_features = np.concatenate([color_features, texture_features])
    return combined_features


def load_image(image_path):
    """
    Load an image from the given path and convert it to RGB.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray or None: The image in RGB format if loaded successfully,
                               otherwise None.
    """
    # Read the image using OpenCV.
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image:", image_path)
        return None
    # Convert the image from BGR (OpenCV default) to RGB.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class LandUseClassifier:
    """
    Basic land use classifier using a Support Vector Classifier (SVC).
    """

    def __init__(self):
        # Initialize an SVC with probability estimates enabled.
        self.clf = SVC(probability=True, random_state=42)

    def train(self, X, y):
        """
        Train the classifier on the provided feature matrix and labels.

        Parameters:
            X (numpy.ndarray): Feature matrix.
            y (numpy.ndarray): Labels corresponding to the feature matrix.
        """
        self.clf.fit(X, y)

    def predict(self, features):
        """
        Predict class labels for the provided features.

        Parameters:
            features (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        return self.clf.predict(features)

    def predict_proba(self, features):
        """
        Predict class probabilities for the provided features.

        Parameters:
            features (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted class probabilities.
        """
        return self.clf.predict_proba(features)

    def evaluate(self, X, y):
        """
        Evaluate the classifier on the provided data and print a report.

        Parameters:
            X (numpy.ndarray): Feature matrix.
            y (numpy.ndarray): True labels.
        """
        y_pred = self.clf.predict(X)
        print("Classification Report:")
        print(classification_report(y, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))

    def save(self, path):
        """
        Save the trained classifier to a file.

        Parameters:
            path (str): File path where the classifier will be saved.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)

    def load(self, path):
        """
        Load a classifier from a file.

        Parameters:
            path (str): File path to load the classifier from.
        """
        with open(path, 'rb') as f:
            self.clf = pickle.load(f)


class RobustLandUseClassifier:
    """
    Enhanced classifier that uses a scikit-learn pipeline for scaling and hyperparameter tuning.

    This classifier builds a pipeline with StandardScaler and SVC, and tunes parameters
    using GridSearchCV.
    """

    def __init__(self):
        # Create a pipeline with a scaler and SVC.
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, random_state=42))
        ])
        self.best_params_ = None

    def train(self, X, y):
        """
        Train the classifier using GridSearchCV for hyperparameter tuning.

        Parameters:
            X (numpy.ndarray): Feature matrix.
            y (numpy.ndarray): Labels corresponding to the feature matrix.
        """
        # Define the parameter grid for hyperparameter tuning.
        param_grid = {
            'svc__C': [0.1, 1, 10],
            'svc__gamma': ['scale', 'auto'],
            'svc__kernel': ['rbf', 'linear']
        }
        # Initialize GridSearchCV with 5-fold cross-validation.
        grid = GridSearchCV(self.pipeline, param_grid, cv=5, scoring='accuracy')
        grid.fit(X, y)
        # Update the pipeline with the best estimator from grid search.
        self.pipeline = grid.best_estimator_
        self.best_params_ = grid.best_params_
        print("Best hyperparameters:", self.best_params_)

    def predict(self, X):
        """
        Predict class labels for the provided features using the tuned pipeline.

        Parameters:
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for the provided features using the tuned pipeline.

        Parameters:
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted class probabilities.
        """
        return self.pipeline.predict_proba(X)

    def evaluate(self, X, y):
        """
        Evaluate the tuned pipeline on the provided data and print a report.

        Parameters:
            X (numpy.ndarray): Feature matrix.
            y (numpy.ndarray): True labels.
        """
        y_pred = self.pipeline.predict(X)
        print("Classification Report:")
        print(classification_report(y, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))

    def save(self, path):
        """
        Save the tuned pipeline to a file.

        Parameters:
            path (str): File path where the pipeline will be saved.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)

    def load(self, path):
        """
        Load a pipeline from a file.

        Parameters:
            path (str): File path to load the pipeline from.
        """
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
