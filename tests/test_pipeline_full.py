"""
test_pipeline_full.py
---------------------
Full pipeline evaluation for land use classification using multiple feature
extraction methods and segmentation techniques.

Author: Robert Wilcox
Date: 03.10.25

This script evaluates three models trained on different feature extraction methods:
    - Raw features extracted from the full image.
    - Aggregated features from K-Means segmentation.
    - Aggregated features from Fuzzy C-Means segmentation.

It builds test datasets from a CSV file, evaluates the models on each dataset, and
plots confusion matrices, overall accuracy, evaluation time comparisons, and how
fuzzy segmentation performance varies with different k values.
"""

import os
import cv2
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import time
from sklearn.metrics import classification_report, confusion_matrix

from src.classification.classifier import (LandUseClassifier, extract_combined_features,
                                           extract_segmented_features)

# ----- Configuration Parameters -----
DEFAULT_K = 2          # Default number of segments for standard evaluation (for k-means & fuzzy default)
M = 2                  # Fuzziness parameter for fuzzy C-means segmentation
MAX_IMAGES = 500       # Maximum number of images to process in the test
FUZZY_K_RANGE = range(1, 6)  # Adjustable range for k values to test in fuzzy segmentation
# --------------------------------------

# Global list to store confusion matrix figures for later display.
confusion_figs = []


def load_image(image_path):
    """
    Load an image from the given path and convert it to RGB.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray or None: The loaded image in RGB format, or None if loading fails.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image:", image_path)
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_features_raw(img_path):
    """
    Extract combined features from the raw (full) image.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        numpy.ndarray or None: Combined features or None if image loading fails.
    """
    img = load_image(img_path)
    if img is None:
        return None
    return extract_combined_features(img)


def extract_features_kmeans(img_path, k=DEFAULT_K, m=M):
    """
    Extract aggregated features using K-Means segmentation.

    Parameters:
        img_path (str): Path to the image file.
        k (int): Number of segments/clusters.
        m (int): Fuzziness parameter (unused for k-means but kept for signature consistency).

    Returns:
        numpy.ndarray or None: Aggregated features from k-means segmentation, or None if loading fails.
    """
    img = load_image(img_path)
    if img is None:
        return None
    return extract_segmented_features(img, k=k, m=m, method='kmeans')


def extract_features_fuzzy(img_path, k=DEFAULT_K, m=M):
    """
    Extract aggregated features using Fuzzy C-Means segmentation.

    Parameters:
        img_path (str): Path to the image file.
        k (int): Number of segments/clusters.
        m (int): Fuzziness parameter.

    Returns:
        numpy.ndarray or None: Aggregated features from fuzzy segmentation, or None if loading fails.
    """
    img = load_image(img_path)
    if img is None:
        return None
    return extract_segmented_features(img, k=k, m=m, method='fuzzy')


def build_test_dataset(test_df):
    """
    Build test datasets for raw, kmeans, and fuzzy features from the test CSV.

    Parameters:
        test_df (pandas.DataFrame): DataFrame loaded from the test CSV.

    Returns:
        tuple: (X_raw, X_kmeans, X_fuzzy, y_test) as numpy arrays.
    """
    X_raw, X_kmeans, X_fuzzy, y_test = [], [], [], []
    start_time = time.time()
    for idx, row in test_df.iterrows():
        if len(y_test) >= MAX_IMAGES:
            break
        img_filename = row["Filename"]
        img_path = os.path.join(os.getcwd(), "data", "raw", "EuroSAT", img_filename)
        feat_raw = extract_features_raw(img_path)
        feat_km = extract_features_kmeans(img_path, k=DEFAULT_K, m=M)
        feat_fuzzy = extract_features_fuzzy(img_path, k=DEFAULT_K, m=M)
        if feat_raw is not None and feat_km is not None and feat_fuzzy is not None:
            X_raw.append(feat_raw)
            X_kmeans.append(feat_km)
            X_fuzzy.append(feat_fuzzy)
            y_test.append(row["ClassName"])
    elapsed = time.time() - start_time
    print(f"Time to build standard test dataset: {elapsed:.2f} seconds")
    return np.array(X_raw), np.array(X_kmeans), np.array(X_fuzzy), np.array(y_test)


def build_test_dataset_fuzzy(test_df, k_val, m=M):
    """
    Build a test dataset for fuzzy segmentation using a specified k value.

    Parameters:
        test_df (pandas.DataFrame): DataFrame loaded from the test CSV.
        k_val (int): The number of clusters for fuzzy segmentation.
        m (int): Fuzziness parameter.

    Returns:
        tuple: (X_fuzzy, y_test) as numpy arrays.
    """
    X_fuzzy, y_test = [], []
    start_time = time.time()
    for idx, row in test_df.iterrows():
        if len(y_test) >= MAX_IMAGES:
            break
        img_filename = row["Filename"]
        img_path = os.path.join(os.getcwd(), "data", "raw", "EuroSAT", img_filename)
        feat_fuzzy = extract_features_fuzzy(img_path, k=k_val, m=m)
        if feat_fuzzy is not None:
            X_fuzzy.append(feat_fuzzy)
            y_test.append(row["ClassName"])
    elapsed = time.time() - start_time
    print(f"Time to build fuzzy test dataset for k={k_val}: {elapsed:.2f} seconds")
    return np.array(X_fuzzy), np.array(y_test)


def load_model(model_filename):
    """
    Load a LandUseClassifier model from a given filename.

    Parameters:
        model_filename (str): Filename of the saved model.

    Returns:
        LandUseClassifier: Loaded classifier.
    """
    model = LandUseClassifier()
    model_path = os.path.join(os.getcwd(), "models", model_filename)
    model.load(model_path)
    return model


def plot_confusion_matrix(cm, classes, title, overall_accuracy, total_samples, per_class_accuracy, cmap=plt.cm.Blues):
    """
    Plot a color-coded confusion matrix with additional information in the title.

    Parameters:
        cm (numpy.ndarray): Confusion matrix.
        classes (list): List of class labels.
        title (str): Title for the plot.
        overall_accuracy (float): Overall accuracy percentage.
        total_samples (int): Total number of samples.
        per_class_accuracy (dict): Dictionary of per-class accuracy percentages.
        cmap: Colormap for the heatmap.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                     xticklabels=classes, yticklabels=classes)
    ax.set_xscale('linear')
    ax.set_yscale('linear')

    per_class_text = " | ".join([f"{cl}: {acc:.1f}%" for cl, acc in per_class_accuracy.items()])
    title_text = (f"{title}\nOverall Accuracy: {overall_accuracy:.2f}% | Total Samples: {total_samples}\n"
                  f"Per-class: {per_class_text}")
    wrapped_title = "\n".join(textwrap.wrap(title_text, width=60))
    plt.title(wrapped_title, fontsize=12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    confusion_figs.append(fig)


def evaluate_model(model, X, y, dataset_name, model_name):
    """
    Evaluate a model on dataset X with labels y.

    Measures prediction time (excluding plotting), prints classification report and accuracy details,
    and stores the confusion matrix figure.

    Parameters:
        model (LandUseClassifier): Classifier to evaluate.
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): True labels.
        dataset_name (str): Name of the test dataset.
        model_name (str): Name of the model being evaluated.

    Returns:
        tuple: (cm, overall_accuracy, elapsed_pred)
            cm: Confusion matrix.
            overall_accuracy (float): Overall accuracy percentage.
            elapsed_pred (float): Time taken for prediction in seconds.
    """
    start_pred = time.time()
    y_pred = model.clf.predict(X)
    elapsed_pred = time.time() - start_pred
    print(f"Prediction time for {model_name} on {dataset_name} dataset: {elapsed_pred:.2f} seconds")

    print(f"=== Evaluation of {model_name} on {dataset_name} features ===")
    print(classification_report(y, y_pred, zero_division=0))

    cm = confusion_matrix(y, y_pred)
    overall_accuracy = np.trace(cm) / np.sum(cm) * 100
    total_samples = np.sum(cm)

    classes = np.unique(y)
    per_class_accuracy = {}
    for i, cl in enumerate(classes):
        if np.sum(cm[i, :]) > 0:
            per_class_accuracy[cl] = cm[i, i] / np.sum(cm[i, :]) * 100
        else:
            per_class_accuracy[cl] = 0.0

    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    print("Per-class accuracy:")
    for cl, acc in per_class_accuracy.items():
        print(f"  {cl}: {acc:.2f}%")
    print("Total samples evaluated:", total_samples)

    plot_confusion_matrix(cm, classes=classes,
                          title=f"{model_name} on {dataset_name} features",
                          overall_accuracy=overall_accuracy,
                          total_samples=total_samples,
                          per_class_accuracy=per_class_accuracy)
    return cm, overall_accuracy, elapsed_pred


def test_pipeline_full():
    """
    Evaluate all three models (trained on raw, kmeans, fuzzy data) on all three test datasets (raw, kmeans, fuzzy).

    Prints detailed classification reports, timing information, and memory usage for each dataset.
    Builds and displays an overall accuracy comparison matrix and an evaluation time matrix.
    Additionally, loops through adjustable k values (FUZZY_K_RANGE) for fuzzy segmentation and plots
    overall accuracy vs. k.
    Finally, after all computations, displays all stored confusion matrices and a timing comparison plot.
    """
    start_total = time.time()

    # Load test CSV.
    csv_path = os.path.join(os.getcwd(), "data", "raw", "EuroSAT", "test.csv")
    test_df = pd.read_csv(csv_path, index_col=0)
    print("Test CSV loaded. Number of test images (max):", MAX_IMAGES)

    # Build standard test datasets for raw, kmeans, and default fuzzy (using DEFAULT_K).
    start_dataset = time.time()
    X_test_raw, X_test_kmeans, X_test_fuzzy, y_test = build_test_dataset(test_df)
    elapsed_dataset = time.time() - start_dataset
    print(f"Time to build standard test datasets: {elapsed_dataset:.2f} seconds")

    print("Test dataset shapes:")
    print("Raw:", X_test_raw.shape)
    print("K-Means:", X_test_kmeans.shape)
    print("Fuzzy (default):", X_test_fuzzy.shape)

    # Print memory usage for each test dataset (in MB).
    print("Memory usage for raw dataset: {:.6f} MB".format(X_test_raw.nbytes / (1024 * 1024)))
    print("Memory usage for K-Means dataset: {:.6f} MB".format(X_test_kmeans.nbytes / (1024 * 1024)))
    print("Memory usage for Fuzzy dataset: {:.6f} MB".format(X_test_fuzzy.nbytes / (1024 * 1024)))

    # Load the three models.
    classifier_raw = load_model("landuse_classifier_raw.pkl")
    classifier_km = load_model("landuse_classifier_kmeans.pkl")
    classifier_fuzzy = load_model("landuse_classifier_fuzzy.pkl")

    # Standard evaluation: Evaluate each model on each test dataset.
    models = [("Raw Model", classifier_raw),
              ("K-Means Model", classifier_km),
              ("Fuzzy Model", classifier_fuzzy)]
    test_datasets = [("Raw", X_test_raw),
                     ("K-Means", X_test_kmeans),
                     ("Fuzzy", X_test_fuzzy)]

    accuracy_matrix = np.zeros((len(models), len(test_datasets)))
    eval_time_matrix = np.zeros((len(models), len(test_datasets)))
    start_eval = time.time()
    for i, (model_name, model) in enumerate(models):
        for j, (dataset_name, X_test) in enumerate(test_datasets):
            _, overall_accuracy, elapsed_eval = evaluate_model(model, X_test, y_test, dataset_name, model_name)
            accuracy_matrix[i, j] = overall_accuracy
            eval_time_matrix[i, j] = elapsed_eval
            print("\n")
    elapsed_eval_total = time.time() - start_eval
    print(f"Time to evaluate all models on standard datasets: {elapsed_eval_total:.2f} seconds")

    std_model_names = [m[0] for m in models]
    std_dataset_names = [d[0] for d in test_datasets]
    acc_df = pd.DataFrame(accuracy_matrix, index=std_model_names, columns=std_dataset_names)
    eval_time_df = pd.DataFrame(eval_time_matrix, index=std_model_names, columns=std_dataset_names)

    # Plot overall accuracy comparison matrix.
    plt.figure(figsize=(8, 6))
    sns.heatmap(acc_df, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Overall Accuracy Comparison Matrix (%)", fontsize=14)
    plt.ylabel("Model Trained On")
    plt.xlabel("Test Dataset")
    plt.tight_layout()
    plt.savefig("overall_accuracy_comparison.png")
    plt.show()

    print("Standard Overall Accuracy Comparison Matrix:")
    print(acc_df)

    # Plot evaluation time comparison matrix.
    plt.figure(figsize=(8, 6))
    sns.heatmap(eval_time_df, annot=True, fmt=".4f", cmap="magma")
    plt.title("Model Evaluation Time Comparison Matrix (seconds)", fontsize=14)
    plt.ylabel("Model Trained On")
    plt.xlabel("Test Dataset")
    plt.tight_layout()
    plt.savefig("evaluation_time_comparison.png")
    plt.show()

    print("Standard Evaluation Time Comparison Matrix:")
    print(eval_time_df)

    # Extra: Loop through adjustable k values for fuzzy segmentation.
    fuzzy_accuracies = []
    fuzzy_eval_times = []
    for k_val in FUZZY_K_RANGE:
        start_fuzzy = time.time()
        X_test_fuzzy_k, y_test_k = build_test_dataset_fuzzy(test_df, k_val, m=M)
        elapsed_fuzzy_build = time.time() - start_fuzzy
        print(f"Time to build fuzzy test dataset for k={k_val}: {elapsed_fuzzy_build:.2f} seconds")
        print(f"Evaluating Fuzzy Model with k = {k_val}")
        start_fuzzy_eval = time.time()
        _, overall_accuracy, elapsed_fuzzy_eval = evaluate_model(classifier_fuzzy, X_test_fuzzy_k, y_test_k, f"Fuzzy (k={k_val})", "Fuzzy Model")
        fuzzy_accuracies.append(overall_accuracy)
        fuzzy_eval_times.append(elapsed_fuzzy_eval)
        print(f"Evaluation time for fuzzy model with k={k_val}: {elapsed_fuzzy_eval:.2f} seconds\n")

    plt.figure(figsize=(8, 6))
    plt.plot(list(FUZZY_K_RANGE), fuzzy_accuracies, marker='o', linestyle='-', color='b')
    plt.title("Fuzzy Model Overall Accuracy vs. k", fontsize=14)
    plt.xlabel("k (Number of Segments)")
    plt.ylabel("Overall Accuracy (%)")
    plt.xticks(list(FUZZY_K_RANGE))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fuzzy_accuracy_vs_k.png")
    plt.show()

    print("Fuzzy Model Overall Accuracies for different k values:")
    for k_val, acc in zip(FUZZY_K_RANGE, fuzzy_accuracies):
        print(f"  k = {k_val}: {acc:.2f}%")

    elapsed_total = time.time() - start_total
    print(f"Total pipeline runtime: {elapsed_total:.2f} seconds")

    # Build a timing comparison bar chart for key stages.
    plt.figure(figsize=(10, 6))
    timing_data = {
        "Test Dataset Build": elapsed_dataset,
        "Standard Model Eval": elapsed_eval_total,
        "Total Pipeline Runtime": elapsed_total
    }
    timing_names = list(timing_data.keys())
    timing_values = list(timing_data.values())
    sns.barplot(x=timing_names, y=timing_values, palette="viridis")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Pipeline Stage")
    plt.title("Timing Comparison for Pipeline Stages", fontsize=14)
    for i, v in enumerate(timing_values):
        plt.text(i, v + 0.01, f"{v:.2f} sec", ha="center", va="bottom", fontsize=12)
    plt.tight_layout()
    plt.savefig("timing_comparison.png")
    plt.show()

    # Finally, display all stored confusion matrix figures.
    for fig in confusion_figs:
        fig.show()


if __name__ == "__main__":
    test_pipeline_full()
