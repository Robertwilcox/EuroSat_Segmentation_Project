"""
main.py
-------
A module for processing images using K-Means segmentation and a pre-trained
land use classifier. Extracts color features from segmented regions and
predicts land use classes.

Author: Robert Wilcox
Date: 03.10.25

This script loads a pre-trained classifier, processes images from a specified
dataset directory, segments the images using K-Means, extracts features,
predicts class probabilities, and logs the top predictions to a CSV file.
It also optionally visualizes segmented images.
"""

import os
import cv2
import numpy as np
from segmentation.kmeans import kmeans_segmentation
from classification.classifier import extract_color_features, LandUseClassifier
import matplotlib.pyplot as plt
import csv


def process_image(image_path, classifier, k=5):
    """
    Process a single image: load, segment using K-Means, extract color features,
    and predict class probabilities for each segmented region.

    Parameters:
        image_path (str): Path to the image file.
        classifier (LandUseClassifier): Pre-trained land use classifier.
        k (int): Number of segments/clusters for K-Means segmentation.

    Returns:
        tuple: (segmented_image, top_classes, original_image)
            segmented_image (numpy.ndarray): Image with segments replaced by centroid colors.
            top_classes (numpy.ndarray or None): Top two predicted classes for the image.
            original_image (numpy.ndarray): The original loaded image in RGB format.
    """
    # Load image using OpenCV.
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return None, None, None
    # Convert the image from BGR (OpenCV default) to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Segment the image using K-Means segmentation.
    labels, centroids = kmeans_segmentation(image, k)
    # Create a segmented image by replacing each pixel with its centroid's color.
    segmented_image = centroids[labels]

    # For each segment (cluster), extract color features.
    features_list = []
    for i in range(k):
        # Create a mask for the current segment.
        mask = (labels == i)
        # Skip if the segment is empty.
        if np.sum(mask) == 0:
            continue
        # Copy the image and zero out pixels not belonging to the segment.
        cluster_img = image.copy()
        cluster_img[~mask] = 0
        # Extract color features from the segmented region.
        features = extract_color_features(cluster_img)
        features_list.append(features)
    features_array = np.array(features_list)

    # Check if any features were extracted.
    if features_array.size == 0:
        print(f"No segments detected for feature extraction in {image_path}")
        return segmented_image, None, image

    # Predict class probabilities for each segmented region.
    segment_predictions = classifier.predict_proba(features_array)
    # Aggregate the probabilities by taking the mean across segments.
    avg_probabilities = np.mean(segment_predictions, axis=0)
    # Get the indices of the top two predicted classes.
    top_indices = np.argsort(avg_probabilities)[-2:][::-1]
    # Retrieve the class labels corresponding to the top indices.
    top_classes = classifier.clf.classes_[top_indices]

    return segmented_image, top_classes, image


def main():
    """
    Main function to process images, predict land use classes, and log results.
    """
    # Define data and model directories.
    data_dir = os.path.join(os.getcwd(), "..", "data", "raw", "EuroSAT")
    model_path = os.path.join(os.getcwd(), "..", "models", "landuse_classifier.pkl")

    # Load the pre-trained classifier.
    classifier = LandUseClassifier()
    try:
        classifier.load(model_path)
        print("Classifier loaded from", model_path)
    except Exception as e:
        print("Failed to load classifier. Make sure you have trained one and saved it at", model_path)
        return

    # Get list of classes (folder names) from the dataset directory.
    classes = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))]

    # Open a CSV file to log the prediction results.
    results_csv = os.path.join(os.getcwd(), "..", "results", "predictions.csv")
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImagePath", "TopClass1", "TopClass2"])

        processed_count = 0
        max_images = 100  # Limit processing to 100 images

        # Loop over each class folder and process the images.
        for cls in classes:
            class_dir = os.path.join(data_dir, cls)
            # Filter for image files with valid extensions.
            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            for img_file in image_files:
                if processed_count >= max_images:
                    break
                img_path = os.path.join(class_dir, img_file)
                segmented_img, top_classes, orig_img = process_image(img_path, classifier, k=5)
                if top_classes is not None:
                    writer.writerow([img_path, top_classes[0], top_classes[1]])
                    print(f"Processed {img_path}: {top_classes}")
                    processed_count += 1

                    # Optionally visualize a sample (e.g., the first image of each class)
                    if processed_count <= len(classes):
                        plt.figure(figsize=(8, 8))
                        plt.imshow(segmented_img.astype(np.uint8))
                        plt.title(f"Segmented {cls} sample")
                        plt.axis("off")
                        # Uncomment the next line to display the image.
                        # plt.show()
            if processed_count >= max_images:
                break

    print("Processing complete. Predictions saved to", results_csv)


if __name__ == "__main__":
    main()
