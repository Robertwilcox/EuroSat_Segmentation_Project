import numpy as np
import cv2

def kmeans_segmentation_lab(image, k, max_iter=100, tol=1e-4):
    """
    Perform K-means segmentation on an input image in Lab color space.
    The algorithm uses only the a* and b* channels for clustering.
    
    Parameters:
      image: Input image as a numpy array in RGB format.
      k: Number of clusters.
      max_iter: Maximum number of iterations.
      tol: Tolerance for convergence.
      
    Returns:
      labels_image: A 2D array of cluster labels with shape (H, W).
      centroids: The final cluster centers in the a*b* space (shape (k, 2)).
      segmented_image_rgb: The segmented image reconstructed in RGB format.
    """
    # Convert image from RGB to Lab color space.
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    # Extract the a* and b* channels (ignore L* for clustering).
    ab = lab_image[:, :, 1:3]
    h, w, _ = ab.shape
    ab_reshaped = ab.reshape(-1, 2).astype(np.float32)
    
    # Initialize centroids randomly from the a*b* pixel values.
    indices = np.random.choice(ab_reshaped.shape[0], k, replace=False)
    centroids = ab_reshaped[indices]
    
    # Iterate the K-means algorithm.
    for i in range(max_iter):
        old_centroids = centroids.copy()
        # Compute Euclidean distances between each pixel and each centroid.
        distances = np.linalg.norm(ab_reshaped[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        # Update each centroid as the mean of the pixels assigned to it.
        for j in range(k):
            if np.any(labels == j):
                centroids[j] = np.mean(ab_reshaped[labels == j], axis=0)
            else:
                centroids[j] = ab_reshaped[np.random.choice(ab_reshaped.shape[0])]
        # Check for convergence.
        diff = np.linalg.norm(centroids - old_centroids)
        if diff < tol:
            #print(f"Converged in {i+1} iterations.")
            break
    
    # Reshape labels to the original image dimensions.
    labels_image = labels.reshape(h, w)
    
    # Reconstruct the segmented image:
    # Replace the a* and b* channels with the cluster centers for each pixel,
    # but keep the original L* channel from the Lab image.
    segmented_lab = lab_image.copy()
    for j in range(k):
        segmented_lab[labels_image == j, 1:3] = centroids[j]
    # Convert the modified Lab image back to RGB.
    segmented_image_rgb = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2RGB)
    
    return labels_image, centroids, segmented_image_rgb

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Update the path below with an actual image file from your dataset.
    img_path = "../../data/raw/EuroSat/AnnualCrop/AnnualCrop_12.jpg"
    image = cv2.imread(img_path)
    if image is None:
        print("Failed to load image. Check the path.")
    else:
        # Convert image from BGR to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        k = 5  # Number of clusters (as in the paper, they group the image into 5 classes)
        labels_image, centroids, segmented_image_rgb = kmeans_segmentation_lab(image, k)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(segmented_image_rgb.astype(np.uint8))
        plt.title("K-Means Segmentation in Lab Space")
        plt.axis('off')
        plt.show()
