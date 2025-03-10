import os
import cv2
import numpy as np
import pytest
import pandas as pd
from src.classification.classifier import extract_color_features, LandUseClassifier

def load_image_and_extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return extract_color_features(img)

def test_feature_extraction():
    img_path = os.path.join(os.getcwd(), "data", "raw", "EuroSAT", "Forest", "Forest_1.jpg")
    features = load_image_and_extract_features(img_path)
    assert features is not None, "Feature extraction failed."
    assert features.shape[0] == 6, "Extracted feature vector length is not 6."

def test_classifier_training_and_prediction():
    csv_path = os.path.join(os.getcwd(), "data", "raw", "EuroSAT", "train.csv")
    df = pd.read_csv(csv_path, index_col=0)
    sample_df = df.sample(n=30, random_state=42)
    
    X = []
    y = []
    for idx, row in sample_df.iterrows():
        img_path = os.path.join(os.getcwd(), "data", "raw", "EuroSAT", row["Filename"])
        features = load_image_and_extract_features(img_path)
        if features is not None:
            X.append(features)
            y.append(row["ClassName"])
    
    X = np.array(X)
    y = np.array(y)
    assert X.shape[0] > 0, "No features extracted."
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    classifier = LandUseClassifier()
    classifier.train(X_train, y_train)
    
    preds = classifier.predict(X_val)
    valid_classes = set(sample_df["ClassName"].unique())
    for p in preds:
        assert p in valid_classes, f"Prediction {p} is not a valid class."

if __name__ == "__main__":
    pytest.main()
