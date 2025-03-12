"""
kmeans.py
---------
A module for performing k-means segmentation on an image.

Author: Robert Wilcox
Date: 03.10.25

This script implements k-means clustering for image segmentation. It
includes functions for initializing centroids, assigning pixels to
the nearest centroids, updating centroid positions, and performing the
complete segmentation process. The code is formatted according to PEP8
standards.
"""

import numpy as np
import cv2


def initialize_centroids(image, k):
    """
    Initialize k centroids by randomly selecting k pixels from the image.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array of shape (H, W, C)
        k (int): Number of clusters

    Returns:
        numpy.ndarray: A numpy array of shape (k, C) containing the initial centroid colors.
    """
    # Get image dimensions (height, width, channels)
    h, w, c = image.shape
    # Reshape image to a list of pixels with color channels.
    pixels = image.reshape(-1, c)
    # Randomly choose k unique indices from the pixels.
    indices = np.random.choice(pixels.shape[0], k, replace=False)
    # Use the chosen indices to select initial centroids and cast them to float32.
    centroids = pixels[indices].astype(np.float32)
    return centroids


def assign_pixels_to_centroids(image, centroids):
    """
    Assign each pixel to the nearest centroid based on Euclidean distance.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array of shape (H, W, C)
        centroids (numpy.ndarray): A numpy array of shape (k, C) containing centroid colors

    Returns:
        numpy.ndarray: A numpy array of shape (H, W) containing the index of the nearest centroid for each pixel.
    """
    # Retrieve image dimensions.
    h, w, c = image.shape
    # Reshape image into a 2D array where each row is a pixel's color.
    pixels = image.reshape(-1, c)
    # Calculate the Euclidean distance between each pixel and each centroid.
    # The expression 'pixels[:, np.newaxis]' reshapes pixels to (num_pixels, 1, c)
    # so that broadcasting can compute distances to each centroid.
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
    # For each pixel, select the index of the centroid with the minimum distance.
    labels = np.argmin(distances, axis=1)
    # Reshape the flat label array back into the original image dimensions.
    return labels.reshape(h, w)


def update_centroids(image, labels, k):
    """
    Update centroid positions by computing the mean color of all pixels assigned to each centroid.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array of shape (H, W, C)
        labels (numpy.ndarray): Label map of shape (H, W) indicating cluster assignment for each pixel
        k (int): Number of clusters

    Returns:
        numpy.ndarray: Updated centroids as a numpy array of shape (k, C)
    """
    # Retrieve image dimensions.
    h, w, c = image.shape
    # Reshape the image into a 2D array where each row represents a pixel.
    pixels = image.reshape(-1, c)
    # Flatten the label matrix to easily index the pixels.
    labels_flat = labels.reshape(-1)
    # Initialize an array to store the updated centroids.
    centroids = np.zeros((k, c), dtype=np.float32)

    # Iterate through each centroid index.
    for i in range(k):
        # Check if there are any pixels assigned to the current centroid.
        if np.any(labels_flat == i):
            # Calculate the mean color for the pixels in the cluster.
            centroids[i] = np.mean(pixels[labels_flat == i], axis=0)
        else:
            # If no pixels are assigned, randomly reinitialize the centroid.
            centroids[i] = pixels[np.random.choice(pixels.shape[0])]
    return centroids


def kmeans_segmentation(image, k, max_iter=100, tol=1e-4):
    """
    Perform k-means segmentation on an image.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array of shape (H, W, C)
        k (int): Number of clusters
        max_iter (int): Maximum number of iterations for convergence
        tol (float): Tolerance threshold to determine convergence based on centroid movement

    Returns:
        tuple: (labels, centroids, segmented_image)
            labels (numpy.ndarray): A label map of shape (H, W) indicating cluster assignment for each pixel.
            centroids (numpy.ndarray): Final centroids as a numpy array of shape (k, C).
            segmented_image (numpy.ndarray): The segmented image constructed by replacing each pixel with its centroid color.
    """
    # Initialize centroids using random pixels from the image.
    centroids = initialize_centroids(image, k)

    for i in range(max_iter):
        # Store a copy of the old centroids for convergence check.
        old_centroids = centroids.copy()
        # Assign each pixel to the nearest centroid.
        labels = assign_pixels_to_centroids(image, centroids)
        # Update centroid positions based on the new assignment.
        centroids = update_centroids(image, labels, k)
        # Compute the total movement of centroids.
        diff = np.linalg.norm(centroids - old_centroids)
        # If the centroids moved less than the tolerance, the algorithm has converged.
        if diff < tol:
            # Optionally, you can print the number of iterations taken to converge.
            # print(f"Converged in {i + 1} iterations.")
            break

    # Create the segmented image by replacing each pixel with its assigned centroid's color.
    segmented_image = centroids[labels]
    return labels, centroids, segmented_image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Path to an example image file; update the path to match your dataset.
    img_path = "../../data/raw/EuroSat/AnnualCrop/AnnualCrop_12.jpg"
    # Read the image using OpenCV.
    image = cv2.imread(img_path)
    if image is None:
        print("Failed to load image. Check the path.")
    else:
        # Convert the image from BGR (OpenCV's default) to RGB for correct color display.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        k = 5  # Number of clusters (adjust this value as needed)

        # Perform k-means segmentation.
        labels, centroids, segmented_image = kmeans_segmentation(image, k)

        # Plot the segmented image.
        plt.figure(figsize=(8, 8))
        plt.imshow(segmented_image.astype(np.uint8))
        plt.title("K-Means Segmentation Result (Original Color Space)")
        plt.axis('off')
        plt.show()
