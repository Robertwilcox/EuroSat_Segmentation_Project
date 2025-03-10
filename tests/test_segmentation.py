import os
import cv2
import numpy as np
import pytest
from src.segmentation.kmeans import kmeans_segmentation

def test_kmeans_output_dimensions():
    # Use a sample image from the dataset.
    img_path = os.path.join(os.getcwd(), "data", "raw", "EuroSAT", "Forest", "Forest_1.jpg")
    image = cv2.imread(img_path)
    assert image is not None, f"Image not found: {img_path}"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    k = 5
    labels, centroids = kmeans_segmentation(image, k)
    
    # Check that the label map has the same height and width as the original image.
    assert labels.shape == image.shape[:2], "Label map dimensions do not match image dimensions."
    
    # Check that there are exactly k centroids.
    assert centroids.shape[0] == k, "Number of centroids is not equal to k."

def test_kmeans_convergence():
    img_path = os.path.join(os.getcwd(), "data", "raw", "EuroSAT", "Forest", "Forest_1.jpg")
    image = cv2.imread(img_path)
    assert image is not None, "Test image not found."
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    k = 5
    labels1, centroids1 = kmeans_segmentation(image, k)
    labels2, centroids2 = kmeans_segmentation(image, k)
    
    diff = np.linalg.norm(centroids1 - centroids2)
    assert diff < 50, "Centroids vary too much between runs, indicating instability."

if __name__ == "__main__":
    pytest.main()
