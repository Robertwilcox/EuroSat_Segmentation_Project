import os
import cv2
import numpy as np
import pandas as pd
import csv
from src.classification.classifier import LandUseClassifier, extract_color_features
from src.segmentation.kmeans import kmeans_segmentation

def process_image(image_path, classifier, k=5):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return None, None, None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run K-Means segmentation
    labels, centroids = kmeans_segmentation(image, k)
    segmented_image = centroids[labels]
    
    # Extract features for each cluster
    features_list = []
    for i in range(k):
        mask = (labels == i)
        if np.sum(mask) == 0:
            continue
        cluster_img = image.copy()
        cluster_img[~mask] = 0
        features = extract_color_features(cluster_img)
        features_list.append(features)
    features_array = np.array(features_list)
    
    if features_array.size == 0:
        print(f"No segments detected for feature extraction in {image_path}")
        return segmented_image, None, image

    # Predict probabilities for each segment
    segment_predictions = classifier.predict_proba(features_array)
    # Aggregate predictions (simple average)
    avg_probabilities = np.mean(segment_predictions, axis=0)
    # Get the top two predicted classes
    top_indices = np.argsort(avg_probabilities)[-2:][::-1]
    top_classes = classifier.clf.classes_[top_indices]
    
    return segmented_image, top_classes, image

def test_pipeline_full():
    """
    Runs the full pipeline on up to 100 images from the validation CSV,
    collects mistakes (where the true label is not among the top 2 predictions),
    and writes those mistakes to an output CSV file.
    """
    csv_path = os.path.join(os.getcwd(), "data", "raw", "EuroSAT", "validation.csv")
    df = pd.read_csv(csv_path, index_col=0)
    
    # Load the classifier
    classifier = LandUseClassifier()
    model_path = os.path.join(os.getcwd(), "models", "landuse_classifier.pkl")
    try:
        classifier.load(model_path)
    except Exception as e:
        print(f"Classifier could not be loaded from {model_path}. Skipping test.")
        return
    
    mistakes = []
    processed_count = 0
    max_images = 100  # Limit processing to 100 images
    
    for idx, row in df.iterrows():
        if processed_count >= max_images:
            break
        true_label = row["ClassName"]
        img_path = os.path.join(os.getcwd(), "data", "raw", "EuroSAT", row["Filename"])
        processed_count += 1
        segmented_image, top_classes, orig_image = process_image(img_path, classifier, k=5)
        if top_classes is None or true_label not in top_classes:
            mistakes.append((img_path, top_classes if top_classes is not None else "No prediction", true_label))
    
    output_file = os.path.join(os.getcwd(), "results", "pipeline_mistakes.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ImagePath", "Predicted Top Classes", "True Label"])
        for entry in mistakes:
            writer.writerow(entry)
    
    print("Pipeline evaluation complete.")
    print(f"Total images evaluated: {processed_count}")
    print(f"Total mistakes: {len(mistakes)}")
    print(f"Mistakes saved to: {output_file}")

if __name__ == "__main__":
    test_pipeline_full()
