import numpy as np
import cv2

def initialize_membership(image, k):
    """
    Randomly initialize the membership matrix for all pixels.
    
    Parameters:
      image: Input image as a numpy array (H, W, C)
      k: Number of clusters
    
    Returns:
      u: Membership matrix of shape (N, k) where N = H*W. Each row sums to 1.
    """
    h, w, c = image.shape
    N = h * w
    u = np.random.rand(N, k)
    # Normalize so each row sums to 1
    u = u / np.sum(u, axis=1, keepdims=True)
    return u

def update_centroids(image, u, m, k):
    """
    Vectorized update of the cluster centers (centroids) using fuzzy memberships.
    
    Parameters:
      image: Input image as a numpy array (H, W, C)
      u: Membership matrix of shape (N, k)
      m: Fuzziness parameter (usually >1, commonly m=2)
      k: Number of clusters
    
    Returns:
      centroids: Updated centroids as a numpy array of shape (k, C)
    """
    h, w, c = image.shape
    N = h * w
    pixels = image.reshape(N, c)  # shape (N, c)
    weights = u ** m             # shape (N, k)
    numerator = np.dot(weights.T, pixels)  # shape (k, c)
    denominator = np.sum(weights, axis=0)[:, None]  # shape (k, 1)
    centroids = numerator / (denominator + 1e-8)
    return centroids

def update_membership(image, centroids, m):
    """
    Vectorized update of the membership matrix based on the current centroids.
    
    Parameters:
      image: Input image as a numpy array (H, W, C)
      centroids: Current centroids as a numpy array (k, C)
      m: Fuzziness parameter
    
    Returns:
      u_new: Updated membership matrix of shape (N, k)
    """
    h, w, c = image.shape
    N = h * w
    pixels = image.reshape(N, c)  # shape (N, c)
    k = centroids.shape[0]
    # Compute distances from all pixels to all centroids (vectorized)
    # dist has shape (N, k)
    dist = np.linalg.norm(pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2) + 1e-8
    exponent = 2.0 / (m - 1)
    # Compute ratios: for each pixel i, for each cluster j and l, compute d(i,j)/d(i,l)
    ratios = dist[:, :, None] / dist[:, None, :]  # shape (N, k, k)
    # Sum over l for each pixel i and cluster j
    u_new = 1.0 / np.sum(ratios ** exponent, axis=2)  # shape (N, k)
    return u_new

def fuzzy_cmeans_segmentation(image, k, m=2, max_iter=100, tol=1e-4):
    """
    Perform Fuzzy C-Means segmentation on an input image.
    
    Parameters:
      image: Input image as a numpy array (H, W, C)
      k: Number of clusters
      m: Fuzziness parameter (default 2)
      max_iter: Maximum number of iterations
      tol: Tolerance for convergence (based on changes in the membership matrix)
    
    Returns:
      labels: Final label map (H, W) assigned by maximum membership
      centroids: Final cluster centers (k, C)
      u: Final membership matrix of shape (N, k)
    """
    h, w, c = image.shape
    N = h * w
    u = initialize_membership(image, k)
    
    for iteration in range(max_iter):
        u_old = u.copy()
        centroids = update_centroids(image, u, m, k)
        u = update_membership(image, centroids, m)
        diff = np.linalg.norm(u - u_old)
        if diff < tol:
            # Uncomment the next line to print convergence information.
            # print(f"Fuzzy C-Means converged in {iteration+1} iterations.")
            break
    labels = np.argmax(u, axis=1).reshape(h, w)
    return labels, centroids, u

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Adjust the path as needed.
    img_path = "../../data/raw/EuroSat/AnnualCrop/AnnualCrop_12.jpg"
    image = cv2.imread(img_path)
    if image is None:
        print("Failed to load image. Check the path.")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        k = 5  # experiment with different values
        labels, centroids, u = fuzzy_cmeans_segmentation(image, k, m=2)
        
        # Create segmented image: each pixel is replaced by its cluster centroid color.
        segmented_image = centroids[labels]
        plt.figure(figsize=(8, 8))
        plt.imshow(segmented_image.astype(np.uint8))
        plt.title("Fuzzy C-Means Segmentation Result")
        plt.axis("off")
        plt.show()
