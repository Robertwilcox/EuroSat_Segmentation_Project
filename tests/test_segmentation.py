"""
test_segmentation.py
--------------------
Unit tests for validating the K-Means segmentation functionality.

Author: Robert Wilcox
Date: 03.10.25

This module contains tests to ensure that the K-Means segmentation algorithm:
    - Returns a label map with the correct dimensions.
    - Produces exactly k centroids.
    - Converges to a stable solution over multiple runs.
"""

import os
import cv2
import numpy as np
import pytest
from src.segmentation.kmeans import kmeans_segmentation


def test_kmeans_output_dimensions():
    """
    Test that the output dimensions of K-Means segmentation match the input image.
    
    Ensures that:
        - The label map has the same height and width as the original image.
        - The number of centroids equals the specified k value.
    """
    # Path to a sample image.
    img_path = os.path.join(os.getcwd(), "data", "raw", "EuroSAT", "Forest", "Forest_1.jpg")
    # Load the image using OpenCV.
    image = cv2.imread(img_path)
    assert image is not None, f"Image not found: {img_path}"
    # Convert the image from BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    k = 5
    # Perform K-Means segmentation.
    labels, centroids = kmeans_segmentation(image, k)
    
    # Check that the label map dimensions match the image dimensions.
    assert labels.shape == image.shape[:2], "Label map dimensions do not match image dimensions."
    
    # Check that the number of centroids equals k.
    assert centroids.shape[0] == k, "Number of centroids is not equal to k."


def test_kmeans_convergence():
    """
    Test that K-Means segmentation converges to a stable solution over multiple runs.
    
    Runs the segmentation twice on the same image and compares the centroids.
    The norm of the difference should be below a specified threshold.
    """
    # Path to a sample image.
    img_path = os.path.join(os.getcwd(), "data", "raw", "EuroSAT", "Forest", "Forest_1.jpg")
    # Load the image using OpenCV.
    image = cv2.imread(img_path)
    assert image is not None, "Test image not found."
    # Convert the image from BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    k = 5
    # Run K-Means segmentation twice.
    labels1, centroids1 = kmeans_segmentation(image, k)
    labels2, centroids2 = kmeans_segmentation(image, k)
    
    # Compute the norm of the difference between the two sets of centroids.
    diff = np.linalg.norm(centroids1 - centroids2)
    # Check if the difference is within an acceptable threshold.
    assert diff < 50, "Centroids vary too much between runs, indicating instability."


if __name__ == "__main__":
    pytest.main()
