Below is the updated README reflecting your current file structure:

---

# EuroSat Segmentation Project

**Author:** Robert Wilcox  
**Date:** 03.10.25

This project implements image segmentation and land use classification using both traditional K-Means and Fuzzy C-Means segmentation techniques. It extracts features from images and uses a Support Vector Classifier (SVC) to predict land use classes. The project includes scripts for segmentation, feature extraction, model training, evaluation, and a full pipeline test. It also contains Jupyter notebooks for interactive experimentation and documentation files outlining the proposal, literature review, and final report.

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
- [Documentation](#documentation)
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
EuroSat_Segmentation_Project/
│
├── .pytest_cache/                 # Pytest cache files
├── data/                          # Dataset directory
│   └── raw/
│       ├── EuroSAT/              # Primary dataset images and CSV files
│       └── EuroSATallBands/       # Alternative dataset with all bands
├── docs/                          # Project documentation
│   ├── final_report.md
│   ├── literature_review.md
│   └── project_proposal.md
├── models/                        # Saved trained models
│   ├── landuse_classifier.pkl
│   ├── landuse_classifier_fuzzy.pkl
│   ├── landuse_classifier_kmeans.pkl
│   └── landuse_classifier_raw.pkl
├── notebooks/                     # Jupyter notebooks for segmentation and classification experiments
│   ├── 02_Segmentation.ipynb
│   └── 03_Classification.ipynb
├── output_images/                 # Directory for storing generated plots and output images
├── src/                           # Source code
│   ├── classification/            # Classification module
│   │   ├── classifier.py
│   │   └── __init__.py
│   ├── segmentation/              # Segmentation module
│   │   ├── fuzzy_cmeans.py
│   │   ├── kmeans.py
│   │   └── __init__.py
│   ├── main.py                    # Main script for processing images and classification
│   └── __init__.py
├── tests/                         # Unit tests for the project
│   ├── conftest.py
│   ├── test_classification.py
│   ├── test_evaluation.py
│   ├── test_pipeline_full.py
│   └── test_segmentation.py
├── setup.py                       # (Empty) Setup file for packaging
├── .gitignore                     # Git ignore file
└── README.md                      # This file
```

- **data/**: Contains the raw dataset images and CSV files for training and testing.
- **docs/**: Contains project documentation including the final report, literature review, and proposal.
- **models/**: Contains saved trained models.
- **notebooks/**: Jupyter notebooks for interactive analysis and experiments.
- **output_images/**: Stores generated plots and output images from evaluations.
- **src/**: Contains the source code for segmentation and classification.
- **tests/**: Unit tests for verifying the functionality of segmentation, classification, and the full pipeline.
- **setup.py**: Setup script (currently empty) for packaging the project.

---

## Installation

1. **Clone the Repository:**

   Open a terminal and run:

   ```bash
   git clone https://github.com/your-username/EuroSat_Segmentation_Project.git
   cd EuroSat_Segmentation_Project
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Required Dependencies:**

   Ensure you have Python 3.7 or above installed. Then install the required packages:

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

   If a `requirements.txt` file is not provided, install the packages individually.

---

## Usage

### Segmentation

- **K-Means Segmentation:**

  The script `src/segmentation/kmeans.py` implements K-Means segmentation. Example usage:

  ```python
  from src.segmentation.kmeans import kmeans_segmentation
  import cv2
  import matplotlib.pyplot as plt

  # Load an image and convert it to RGB
  image = cv2.imread("path/to/image.jpg")
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Perform segmentation (e.g., with 5 clusters)
  labels, centroids, segmented_image = kmeans_segmentation(image, k=5)

  # Display the segmented image
  plt.imshow(segmented_image.astype('uint8'))
  plt.title("K-Means Segmentation Result")
  plt.axis('off')
  plt.show()
  ```

- **Fuzzy C-Means Segmentation:**

  Use `src/segmentation/fuzzy_cmeans.py` for fuzzy segmentation:

  ```python
  from src.segmentation.fuzzy_cmeans import fuzzy_cmeans_segmentation

  labels, centroids, u = fuzzy_cmeans_segmentation(image, k=5, m=2)
  ```

### Classification

- **Feature Extraction & Model Training:**

  Use functions in `src/classification/classifier.py` to extract features and train a classifier:

  ```python
  from src/classification/classifier import extract_combined_features, LandUseClassifier

  # Extract features from an image
  features = extract_combined_features(image)

  # Initialize and train the classifier
  classifier = LandUseClassifier()
  classifier.train(X, y)  # where X is the feature matrix and y are the labels

  # Save the trained model
  classifier.save("models/landuse_classifier.pkl")
  ```

- **Running the Main Pipeline:**

  The main pipeline is in `src/main.py`, which processes images, performs segmentation, extracts features, and outputs predictions.

  ```bash
  python src/main.py
  ```

  This script:
  - Loads images from `data/raw/EuroSAT/`
  - Performs K-Means segmentation
  - Extracts color features from segments
  - Predicts class probabilities using a pre-trained classifier
  - Writes predictions to `results/predictions.csv` (or an output file in the appropriate folder)

### Full Pipeline Evaluation

For a complete end-to-end evaluation (including building test datasets, evaluating multiple models, and generating plots), run:

```bash
python tests/test_pipeline_full.py
```

This script:
- Loads a test CSV from the dataset.
- Builds test datasets using raw, K-Means, and fuzzy feature extraction.
- Loads pre-trained classifiers from the `models/` directory.
- Evaluates the models and prints detailed classification reports and confusion matrices.
- Generates plots for accuracy and evaluation time comparisons.
- Iterates through different k values for fuzzy segmentation to study performance variations.

---

## Running Tests

The tests are written using `pytest` and cover various aspects of the project:

- **Feature Extraction and Classification:** `tests/test_classification.py`
- **Evaluation Metrics:** `tests/test_evaluation.py`
- **Segmentation Functionality:** `tests/test_segmentation.py`
- **Full Pipeline Evaluation:** `tests/test_pipeline_full.py`

To run all tests, execute:

```bash
pytest
```

Or run a specific test file:

```bash
pytest tests/test_classification.py
```

---

## Documentation

Project documentation is provided in the `docs/` directory, including:
- **final_report.md:** Final project report.
- **literature_review.md:** Literature review.
- **project_proposal.md:** Project proposal.

---

## Future Work

- **Improve Segmentation:**  
  Explore advanced segmentation techniques (e.g., deep learning-based methods) for more precise region proposals.

- **Model Enhancements:**  
  Experiment with more advanced classifiers or deep learning models for improved land use classification.

- **Data Augmentation:**  
  Integrate data augmentation to enhance model robustness.

- **Integration:**  
  Develop a web interface or API for real-time land use prediction.

---
