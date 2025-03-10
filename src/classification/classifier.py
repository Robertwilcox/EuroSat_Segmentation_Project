import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import local_binary_pattern
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def extract_color_features(image):
    """
    Extract basic color features from an image.
    Computes the mean and standard deviation for each RGB channel.
    
    Parameters:
      image: numpy array of shape (H, W, 3) in RGB format.
    
    Returns:
      features: 1D numpy array containing the means and standard deviations (length 6).
    """
    means = np.mean(image, axis=(0, 1))
    stds = np.std(image, axis=(0, 1))
    return np.concatenate([means, stds])

def extract_texture_features(image, P=8, R=1.0):
    """
    Extract texture features using Local Binary Patterns (LBP).
    Converts the image to grayscale, computes the LBP, and returns a normalized histogram.
    
    Parameters:
      image: numpy array of shape (H, W, 3) in RGB format.
      P: Number of circularly symmetric neighbour set points.
      R: Radius of circle.
      
    Returns:
      lbp_hist: 1D numpy array representing the normalized LBP histogram with a fixed length.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Compute the LBP image
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    # Set a fixed number of bins; for uniform LBP with P=8, 10 bins is common.
    n_bins = 10
    # Compute the histogram of LBP values using the fixed number of bins
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_combined_features(image):
    """
    Combine color and texture features into a single feature vector.
    
    Parameters:
      image: numpy array of shape (H, W, 3) in RGB format.
    
    Returns:
      combined_features: 1D numpy array containing both color and texture features.
    """
    color_features = extract_color_features(image)  # length 6
    texture_features = extract_texture_features(image)  # length depends on LBP configuration (typically 10 for uniform LBP with P=8)
    combined_features = np.concatenate([color_features, texture_features])
    return combined_features

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

class RobustLandUseClassifier:
    """
    Enhanced classifier that uses a scikit-learn pipeline for scaling and hyperparameter tuning.
    This classifier builds a pipeline with StandardScaler and SVC, and tunes parameters using GridSearchCV.
    """
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, random_state=42))
        ])
        self.best_params_ = None
    
    def train(self, X, y):
        param_grid = {
            'svc__C': [0.1, 1, 10],
            'svc__gamma': ['scale', 'auto'],
            'svc__kernel': ['rbf', 'linear']
        }
        grid = GridSearchCV(self.pipeline, param_grid, cv=5, scoring='accuracy')
        grid.fit(X, y)
        self.pipeline = grid.best_estimator_
        self.best_params_ = grid.best_params_
        print("Best hyperparameters:", self.best_params_)
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X, y):
        y_pred = self.pipeline.predict(X)
        print("Classification Report:")
        print(classification_report(y, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
