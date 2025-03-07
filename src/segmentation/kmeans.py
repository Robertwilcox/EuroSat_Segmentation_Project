import numpy as np
import cv2

def initialize_centroids(image, k):
    """
    Initialize k centroids by randomly selecting k pixels from the image.
    
    Parameters:
    - image: Input image as a numpy array of shape (H, W, C)
    - k: Number of clusters
    
    Returns:
    - centroids: A numpy array of shape (k, C) with initial centroid colors.
    """
    h, w, c = image.shape
    pixels = image.reshape(-1, c)
    indices = np.random.choice(pixels.shape[0], k, replace=False)
    centroids = pixels[indices].astype(np.float32)
    return centroids

def assign_pixels_to_centroids(image, centroids):
    """
    Assign each pixel to the nearest centroid based on Euclidean distance.
    
    Parameters:
    - image: Input image as a numpy array of shape (H, W, C)
    - centroids: A numpy array of shape (k, C)
    
    Returns:
    - labels: A numpy array of shape (H, W) containing the index of the nearest centroid for each pixel.
    """
    h, w, c = image.shape
    pixels = image.reshape(-1, c)
    # Compute distances from every pixel to each centroid
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels.reshape(h, w)

def update_centroids(image, labels, k):
    """
    Update centroid positions by computing the mean color of all pixels assigned to each centroid.
    
    Parameters:
    - image: Input image as a numpy array of shape (H, W, C)
    - labels: Label map of shape (H, W)
    - k: Number of clusters
    
    Returns:
    - centroids: Updated centroids as a numpy array of shape (k, C)
    """
    h, w, c = image.shape
    pixels = image.reshape(-1, c)
    labels_flat = labels.reshape(-1)
    centroids = np.zeros((k, c), dtype=np.float32)
    
    for i in range(k):
        if np.any(labels_flat == i):
            centroids[i] = np.mean(pixels[labels_flat == i], axis=0)
        else:
            # If no pixels are assigned, reinitialize the centroid randomly
            centroids[i] = pixels[np.random.choice(pixels.shape[0])]
    return centroids

def kmeans_segmentation(image, k, max_iter=100, tol=1e-4):
    """
    Perform K-Means segmentation on an input image.
    
    Parameters:
    - image: Input image as a numpy array (H, W, C)
    - k: Number of clusters
    - max_iter: Maximum number of iterations for convergence
    - tol: Tolerance value for convergence check (based on centroid movement)
    
    Returns:
    - labels: The final label map (H, W) for the image.
    - centroids: The final centroids (k, C).
    """
    centroids = initialize_centroids(image, k)
    for i in range(max_iter):
        old_centroids = centroids.copy()
        labels = assign_pixels_to_centroids(image, centroids)
        centroids = update_centroids(image, labels, k)
        diff = np.linalg.norm(centroids - old_centroids)
        if diff < tol:
            print(f"Converged in {i+1} iterations.")
            break
    return labels, centroids

if __name__ == "__main__":
    # For testing purposes
    import matplotlib.pyplot as plt
    
    # Update the path below with an actual image file from your dataset
    img_path = "../../data/raw/EuroSat/AnnualCrop/AnnualCrop_12.jpg"
    image = cv2.imread(img_path)
    if image is None:
        print("Failed to load image. Check the path.")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        k = 5  # Number of clusters (you can experiment with different values)
        labels, centroids = kmeans_segmentation(image, k)
        
        # Create the segmented image by replacing each pixel with its centroid color
        segmented_image = centroids[labels]
        plt.figure(figsize=(8, 8))
        plt.imshow(segmented_image.astype(np.uint8))
        plt.title("K-Means Segmentation Result")
        plt.axis('off')
        plt.show()
