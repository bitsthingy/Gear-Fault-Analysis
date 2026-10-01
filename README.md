# Gear Fault Classification from Vibration Signals

Classifying six gear conditions (one healthy, five defective) from two-axis vibration signals using classical machine learning.

## Main Idea

Slice raw vibration signals into short windows, extract simple statistical features (mean, std, RMS, peak, skewness, kurtosis) from each axis, and train classifiers to identify the gear condition. The split is done **chronologically per run, with a gap between partitions**, so overlapping windows never leak between train, validation and test.

## Dataset

- [Mechanical Gear Vibration](https://www.kaggle.com/datasets/hieudaotrung/gear-vibration) (Kaggle)
- 6 classes: No fault, Root crack, Chipped tooth, Eccentricity, Missing tooth, Surface defect
- 150,000 samples per class, 2 sensor channels, 6 operating runs per class

## Approach

| Step | Details |
|------|---------|
| Windowing | 1024-sample windows, 512 hop |
| Features | 12 per window (6 statistics × 2 axes) |
| Split | Per-run chronological 60/20/20 (train/val/test) with a 1-window gap at each boundary |
| Leakage checks | No raw sample overlap across partitions, no duplicate feature vectors across partitions |
| Models | Logistic Regression, Linear SVM, Decision Tree, Random Forest, XGBoost |
| Selection | Validation macro-F1 only (test set untouched until the final comparison) |
| Interpretation | Permutation importance and feature distributions |

## Results

| Model | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|-------|--------------|---------------|---------------|
| **XGBoost** | 0.972 | **0.977** | **0.977** |
| Random Forest (selected on validation) | **0.977** | 0.972 | 0.972 |
| Decision Tree | 0.954 | 0.912 | 0.912 |
| Linear SVM | 0.716 | 0.769 | 0.751 |
| Logistic Regression | 0.713 | 0.759 | 0.745 |

- Tree-based models reach about 97% while linear models stay near 75%, so the class boundaries are non-linear in this feature space.
- Per-class F1 for the selected Random Forest is 0.95 to 0.99 across all six conditions.
- `x_mean` is the most important feature, followed by `y_rms` and `y_mean`. Strong importance of the mean may reflect acquisition differences between recordings rather than gear damage, so this is a within-recording benchmark, not a test on independent machines.

## Run It

```bash
pip install numpy pandas scipy scikit-learn xgboost matplotlib kagglehub
```

Open `Gear_Fault_Classification.ipynb` (built for Google Colab). It needs a `kaggle.json` API key to download the dataset. Outputs (metrics CSVs and plots) are saved to `results/`.

## Tech Stack

Python · scikit-learn · XGBoost · SciPy · pandas · matplotlib
