import numpy as np
import cv2

def initialize_membership(image, k):
    """
    Randomly initialize the membership matrix for all pixels.
    
    Parameters:
    - image: Input image as a numpy array (H, W, C)
    - k: Number of clusters
    
    Returns:
    - u: Membership matrix of shape (N, k) where N = H*W. Each row sums to 1.
    """
    h, w, c = image.shape
    N = h * w
    u = np.random.rand(N, k)
    # Normalize so each row sums to 1
    u = u / np.sum(u, axis=1, keepdims=True)
    return u

def update_centroids(image, u, m, k):
    """
    Update the cluster centers (centroids) using fuzzy memberships.
    
    Parameters:
    - image: Input image as a numpy array (H, W, C)
    - u: Membership matrix of shape (N, k)
    - m: Fuzziness parameter (usually >1, commonly m=2)
    - k: Number of clusters
    
    Returns:
    - centroids: Updated centroids of shape (k, C)
    """
    h, w, c = image.shape
    N = h * w
    pixels = image.reshape(N, c)
    centroids = np.zeros((k, c), dtype=np.float32)
    
    for j in range(k):
        # Numerator: weighted sum of pixels (weights are u^m)
        numerator = np.sum((u[:, j] ** m).reshape(-1, 1) * pixels, axis=0)
        denominator = np.sum(u[:, j] ** m)
        centroids[j] = numerator / (denominator + 1e-8)  # avoid division by zero
    return centroids

def update_membership(image, centroids, m):
    """
    Update the membership matrix based on the current centroids.
    
    Parameters:
    - image: Input image as a numpy array (H, W, C)
    - centroids: Current centroids as a numpy array (k, C)
    - m: Fuzziness parameter
    
    Returns:
    - u_new: Updated membership matrix of shape (N, k)
    """
    h, w, c = image.shape
    N = h * w
    pixels = image.reshape(N, c)
    k = centroids.shape[0]
    u_new = np.zeros((N, k), dtype=np.float32)
    
    for i in range(N):
        for j in range(k):
            # Compute Euclidean distance from pixel i to centroid j
            d_ij = np.linalg.norm(pixels[i] - centroids[j]) + 1e-8  # add a small epsilon to avoid zero
            denom_sum = 0
            for l in range(k):
                d_il = np.linalg.norm(pixels[i] - centroids[l]) + 1e-8
                denom_sum += (d_ij / d_il) ** (2 / (m - 1))
            u_new[i, j] = 1.0 / denom_sum
    return u_new

def fuzzy_cmeans_segmentation(image, k, m=2, max_iter=100, tol=1e-4):
    """
    Perform Fuzzy C-Means segmentation on an input image.
    
    Parameters:
    - image: Input image as a numpy array (H, W, C)
    - k: Number of clusters
    - m: Fuzziness parameter (default 2)
    - max_iter: Maximum number of iterations
    - tol: Tolerance for convergence (based on changes in the membership matrix)
    
    Returns:
    - labels: Final label map (H, W) assigned by maximum membership
    - centroids: Final cluster centers (k, C)
    - u: Final membership matrix of shape (N, k)
    """
    h, w, c = image.shape
    N = h * w
    # Step 1: Initialize membership matrix randomly
    u = initialize_membership(image, k)
    
    for iteration in range(max_iter):
        u_old = u.copy()
        # Step 2: Update centroids
        centroids = update_centroids(image, u, m, k)
        # Step 3: Update membership matrix
        u = update_membership(image, centroids, m)
        # Check for convergence based on membership matrix change
        diff = np.linalg.norm(u - u_old)
        if diff < tol:
            print(f"Fuzzy C-Means converged in {iteration+1} iterations.")
            break
    # Assign each pixel the cluster index with maximum membership
    labels = np.argmax(u, axis=1).reshape(h, w)
    return labels, centroids, u

if __name__ == "__main__":
    # For testing purposes
    import matplotlib.pyplot as plt

    # Adjust the path as needed. Since this script is in src/segmentation,
    # we need to go up two directories to reach the project root.
    img_path = "../../data/raw/EuroSAT/AnnualCrop/AnnualCrop_12.jpg"
    image = cv2.imread(img_path)
    if image is None:
        print("Failed to load image. Check the path.")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        k = 5  # Number of clusters (experiment with different values)
        labels, centroids, u = fuzzy_cmeans_segmentation(image, k, m=2)
        
        # Create segmented image: each pixel gets replaced by its cluster's centroid color.
        segmented_image = centroids[labels]
        plt.figure(figsize=(8, 8))
        plt.imshow(segmented_image.astype(np.uint8))
        plt.title("Fuzzy C-Means Segmentation Result")
        plt.axis("off")
        plt.show()
