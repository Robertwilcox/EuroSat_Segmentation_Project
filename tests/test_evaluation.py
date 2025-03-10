import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

def test_confusion_matrix():
    y_true = np.array(["Forest", "Highway", "Forest", "Residential", "SeaLake", "Forest"])
    y_pred = np.array(["Forest", "Forest", "Forest", "Residential", "SeaLake", "Highway"])
    
    labels = ["Forest", "Highway", "Residential", "SeaLake"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Print the computed confusion matrix
    print("Computed Confusion Matrix:")
    print(cm)
    
    expected_cm = np.array([
        [2, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    np.testing.assert_array_equal(cm, expected_cm)

if __name__ == "__main__":
    pytest.main()
