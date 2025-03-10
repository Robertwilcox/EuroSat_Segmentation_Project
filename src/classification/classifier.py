import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def extract_color_features(image):
    """
    Extract color features from an image.
    Computes the mean and standard deviation for each channel.
    
    Parameters:
      image: numpy array of shape (H, W, C)
    
    Returns:
      features: 1D numpy array containing the means and standard deviations.
    """
    means = np.mean(image, axis=(0, 1))
    stds = np.std(image, axis=(0, 1))
    return np.concatenate([means, stds])

def load_image(image_path):
    """
    Load an image from the given path and convert it to RGB.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image:", image_path)
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class LandUseClassifier:
    def __init__(self):
        self.clf = SVC(probability=True, random_state=42)
    
    def train(self, X, y):
        """
        Train the classifier on the provided feature matrix and labels.
        """
        self.clf.fit(X, y)
    
    def predict(self, features):
        """
        Predict the class labels for the provided features.
        """
        return self.clf.predict(features)
    
    def predict_proba(self, features):
        """
        Predict class probabilities for the provided features.
        """
        return self.clf.predict_proba(features)
    
    def evaluate(self, X, y):
        """
        Evaluate the classifier on the provided data and print a report.
        """
        y_pred = self.clf.predict(X)
        print("Classification Report:")
        print(classification_report(y, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))
    
    def save(self, path):
        """
        Save the trained classifier to a file.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
    
    def load(self, path):
        """
        Load a classifier from a file.
        """
        with open(path, 'rb') as f:
            self.clf = pickle.load(f)
