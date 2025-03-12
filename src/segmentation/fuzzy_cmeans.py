"""
fuxxy_cmeans.py
---------------
A module for performing Fuzzy C-Means segmentation on an image.

Author: Robert Wilcox
Date: 03.10.25

This script implements Fuzzy C-Means clustering for image segmentation.
It includes functions for initializing the membership matrix, updating
centroids using fuzzy memberships, updating the membership matrix based
on current centroids, and performing the complete segmentation process.
The code adheres to PEP8 standards.
"""

import numpy as np
import cv2


def initialize_membership(image, k):
    """
    Randomly initialize the membership matrix for all pixels.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array (H, W, C)
        k (int): Number of clusters

    Returns:
        numpy.ndarray: Membership matrix of shape (N, k) where N = H*W.
                       Each row sums to 1.
    """
    # Get the dimensions of the image.
    h, w, c = image.shape
    # Total number of pixels.
    N = h * w
    # Initialize the membership matrix with random values.
    u = np.random.rand(N, k)
    # Normalize each row so that the sum of memberships for each pixel is 1.
    u = u / np.sum(u, axis=1, keepdims=True)
    return u


def update_centroids(image, u, m, k):
    """
    Vectorized update of the cluster centers (centroids) using fuzzy memberships.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array (H, W, C)
        u (numpy.ndarray): Membership matrix of shape (N, k)
        m (float): Fuzziness parameter (usually >1, commonly m=2)
        k (int): Number of clusters

    Returns:
        numpy.ndarray: Updated centroids as a numpy array of shape (k, C)
    """
    # Retrieve image dimensions.
    h, w, c = image.shape
    # Total number of pixels.
    N = h * w
    # Reshape the image to a 2D array where each row is a pixel.
    pixels = image.reshape(N, c)  # shape (N, c)
    # Compute weights by raising membership values to the power of m.
    weights = u ** m  # shape (N, k)
    # Numerator: weighted sum of pixel values for each cluster.
    numerator = np.dot(weights.T, pixels)  # shape (k, c)
    # Denominator: sum of weights for each cluster.
    denominator = np.sum(weights, axis=0)[:, None]  # shape (k, 1)
    # Compute updated centroids while adding a small constant to avoid division by zero.
    centroids = numerator / (denominator + 1e-8)
    return centroids


def update_membership(image, centroids, m):
    """
    Vectorized update of the membership matrix based on the current centroids.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array (H, W, C)
        centroids (numpy.ndarray): Current centroids as a numpy array (k, C)
        m (float): Fuzziness parameter

    Returns:
        numpy.ndarray: Updated membership matrix of shape (N, k)
    """
    # Retrieve image dimensions.
    h, w, c = image.shape
    # Total number of pixels.
    N = h * w
    # Reshape image into a 2D array where each row is a pixel.
    pixels = image.reshape(N, c)  # shape (N, c)
    # Number of clusters derived from centroids.
    k = centroids.shape[0]
    # Compute the Euclidean distance between each pixel and each centroid.
    # The resulting 'dist' array has shape (N, k).
    dist = np.linalg.norm(pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2) + 1e-8
    # Calculate the exponent used in membership update.
    exponent = 2.0 / (m - 1)
    # Compute the ratio for each pixel: for every pixel i and clusters j and l,
    # calculate the ratio d(i, j) / d(i, l), resulting in shape (N, k, k).
    ratios = dist[:, :, None] / dist[:, None, :]  # shape (N, k, k)
    # Sum the ratios raised to the calculated exponent along the third dimension.
    # Compute the new membership values.
    u_new = 1.0 / np.sum(ratios ** exponent, axis=2)  # shape (N, k)
    return u_new


def fuzzy_cmeans_segmentation(image, k, m=2, max_iter=100, tol=1e-4):
    """
    Perform Fuzzy C-Means segmentation on an input image.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array (H, W, C)
        k (int): Number of clusters
        m (float): Fuzziness parameter (default 2)
        max_iter (int): Maximum number of iterations
        tol (float): Tolerance for convergence (based on changes in the membership matrix)

    Returns:
        tuple:
            labels (numpy.ndarray): Final label map (H, W) assigned by maximum membership
            centroids (numpy.ndarray): Final cluster centers (k, C)
            u (numpy.ndarray): Final membership matrix of shape (N, k)
    """
    # Get image dimensions.
    h, w, c = image.shape
    # Total number of pixels.
    N = h * w
    # Initialize the membership matrix.
    u = initialize_membership(image, k)
    
    # Iterate until convergence or maximum iterations reached.
    for iteration in range(max_iter):
        # Store the old membership matrix to check for convergence.
        u_old = u.copy()
        # Update centroids using current membership values.
        centroids = update_centroids(image, u, m, k)
        # Update membership matrix using the new centroids.
        u = update_membership(image, centroids, m)
        # Calculate the change in membership matrix.
        diff = np.linalg.norm(u - u_old)
        # Check for convergence based on the tolerance.
        if diff < tol:
            # Uncomment the next line to print convergence information.
            # print(f"Fuzzy C-Means converged in {iteration+1} iterations.")
            break
    # Determine the final label for each pixel by taking the cluster with maximum membership.
    labels = np.argmax(u, axis=1).reshape(h, w)
    return labels, centroids, u


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Path to the input image; update this path as necessary.
    img_path = "../../data/raw/EuroSat/AnnualCrop/AnnualCrop_12.jpg"
    # Load the image using OpenCV.
    image = cv2.imread(img_path)
    if image is None:
        print("Failed to load image. Check the path.")
    else:
        # Convert the image from BGR (OpenCV's default) to RGB for correct display.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        k = 5  # Number of clusters (feel free to experiment with different values)
        # Perform Fuzzy C-Means segmentation.
        labels, centroids, u = fuzzy_cmeans_segmentation(image, k, m=2)
        
        # Create a segmented image by replacing each pixel with its cluster centroid color.
        segmented_image = centroids[labels]
        # Plot the resulting segmented image.
        plt.figure(figsize=(8, 8))
        plt.imshow(segmented_image.astype(np.uint8))
        plt.title("Fuzzy C-Means Segmentation Result")
        plt.axis("off")
        plt.show()
