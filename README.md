# Land Use Classification Project

**Author:** Robert Wilcox  
**Date:** 03.10.25

This project implements image segmentation and land use classification using both traditional K-Means and Fuzzy C-Means segmentation techniques. It extracts features from images and uses a Support Vector Classifier (SVC) to predict land use classes. The project includes scripts for segmentation, feature extraction, model training, evaluation, and a full pipeline test.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Segmentation](#segmentation)
  - [Classification](#classification)
  - [Full Pipeline Evaluation](#full-pipeline-evaluation)
- [Running Tests](#running-tests)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

This project provides two main functionalities:

1. **Segmentation:**  
   - **K-Means Segmentation:** Clusters image pixels based on color similarity.
   - **Fuzzy C-Means Segmentation:** Clusters pixels allowing fuzzy (soft) assignments.

2. **Classification:**  
   - Extracts color and texture features from images (both raw and segmented).
   - Trains a land use classifier using scikit-learn’s SVC.
   - Supports an enhanced classifier with a hyperparameter tuning pipeline.

The scripts in the project support building test datasets, evaluating models, and visualizing results (including confusion matrices and evaluation timings).

---

## Project Structure

```
├── README.md
├── segmentation
│   ├── __init__.py
│   ├── kmeans.py
│   ├── fuzzy_cmeans.py
│   └── segmentation_utils.py      # (optional utility file for shared segmentation code)
├── classification
│   ├── __init__.py
│   └── classifier.py
├── tests
│   ├── test_classification.py
│   ├── test_evaluation.py
│   ├── test_segmentation.py
│   └── test_pipeline_full.py
├── data
│   └── raw
│       └── EuroSAT
│           ├── test.csv
│           ├── train.csv
│           └── [land use image folders: Forest, Highway, Residential, SeaLake, etc.]
├── models
│   ├── landuse_classifier.pkl
│   ├── landuse_classifier_raw.pkl
│   ├── landuse_classifier_kmeans.pkl
│   └── landuse_classifier_fuzzy.pkl
└── results
    └── predictions.csv
```

- **segmentation/**: Contains the segmentation algorithms.
- **classification/**: Contains the feature extraction functions and classifiers.
- **tests/**: Unit tests to validate segmentation, classification, evaluation, and the full pipeline.
- **data/raw/EuroSAT/**: Directory with image dataset and CSV files listing image filenames and class labels.
- **models/**: Directory where trained model files are stored.
- **results/**: Directory to store output prediction CSV files and generated plots.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/land-use-classification.git
   cd land-use-classification
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Required Dependencies:**

   Ensure you have Python 3.7 or above installed. Then install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *Dependencies include:*  
   - numpy  
   - opencv-python  
   - scikit-learn  
   - matplotlib  
   - seaborn  
   - pandas  
   - pytest  
   - scikit-image

   If a `requirements.txt` file is not present, you can install the packages individually.

---

## Usage

### Segmentation

- **K-Means Segmentation:**

  The script `segmentation/kmeans.py` implements K-Means segmentation. To use it:

  ```python
  from segmentation.kmeans import kmeans_segmentation
  import cv2
  import matplotlib.pyplot as plt

  # Load an image (ensure it's in RGB)
  image = cv2.imread("path/to/image.jpg")
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Run segmentation (e.g., with 5 clusters)
  labels, centroids, segmented_image = kmeans_segmentation(image, k=5)

  # Visualize the segmented image
  plt.imshow(segmented_image.astype('uint8'))
  plt.title("K-Means Segmentation Result")
  plt.axis('off')
  plt.show()
  ```

- **Fuzzy C-Means Segmentation:**

  Similarly, use `segmentation/fuzzy_cmeans.py` for fuzzy segmentation:

  ```python
  from segmentation.fuzzy_cmeans import fuzzy_cmeans_segmentation

  labels, centroids, u = fuzzy_cmeans_segmentation(image, k=5, m=2)
  ```

### Classification

- **Feature Extraction & Model Training:**

  Use the functions in `classification/classifier.py` to extract features and train a classifier:

  ```python
  from classification.classifier import extract_color_features, LandUseClassifier, extract_combined_features

  # Extract features from an image
  features = extract_combined_features(image)

  # Initialize classifier and train with feature matrix X and labels y
  classifier = LandUseClassifier()
  classifier.train(X, y)

  # Save the trained model
  classifier.save("models/landuse_classifier.pkl")
  ```

- **Running the Main Pipeline:**

  The `main.py` script in the root directory processes images, performs segmentation, extracts features, and outputs predictions to a CSV file.

  ```bash
  python main.py
  ```

  This script will:
  - Load images from `data/raw/EuroSAT/`
  - Perform K-Means segmentation on each image.
  - Extract color features from each segment.
  - Predict class probabilities using a pre-trained classifier.
  - Write predictions to `results/predictions.csv`.

### Full Pipeline Evaluation

For a complete end-to-end evaluation (including building test datasets, evaluating multiple models, and generating plots), run:

```bash
python tests/test_pipeline_full.py
```

This script:
- Loads a test CSV from the dataset.
- Builds test datasets using raw, K-Means, and fuzzy feature extraction.
- Loads pre-trained classifiers.
- Evaluates the models, prints detailed classification reports, and displays confusion matrices.
- Generates accuracy and evaluation time comparison plots.
- Iterates through different k values for fuzzy segmentation to study performance.

---

## Running Tests

Tests are written using `pytest` and cover various aspects of the project:

- **Feature Extraction and Classification:** `tests/test_classification.py`
- **Evaluation Metrics:** `tests/test_evaluation.py`
- **Segmentation Functionality:** `tests/test_segmentation.py`
- **Full Pipeline Evaluation:** `tests/test_pipeline_full.py`

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_classification.py
```

---

## Future Work

- **Improve Segmentation:**  
  Explore advanced segmentation techniques (e.g., DeepLab, U-Net) for better region proposals.

- **Model Enhancements:**  
  Experiment with more advanced classifiers or deep learning models for land use classification.

- **Data Augmentation:**  
  Incorporate data augmentation techniques to improve model robustness.

- **Integration:**  
  Develop a web interface or API for real-time land use prediction.

---