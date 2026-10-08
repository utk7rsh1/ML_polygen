# Machine Learning Assignment: Polynomial Regression

**Student Roll Number:** `BT2024119`  
**GitHub Repository:** [https://github.com/utk7rsh1/ML_polygen](https://github.com/utk7rsh1/ML_polygen)

---

## Overview

This repository contains the training and inference code for Assignment 1 on Polynomial Regression. The assignment consists of two regression problems:
1. **Phase 1: Power Plant Steam Turbine Optimization (`var1`)**
   - 6 operational parameters ($x_1, \dots, x_6$).
   - Target: Net Power Score ($y$).
   - **Chosen Degree:** **Degree 5** with Lasso regularization ($\alpha \approx 0.00796$).
   - **Cross-Validation Score:** MSE = 0.3163, $R^2 = 0.9658$.

2. **Phase 2: Subterranean Thermal Reservoir Mapping (`var2`)**
   - 3 spatial coordinate offsets ($x_1, x_2, x_3$).
   - Target: Thermal Anomaly Score ($y$).
   - **Chosen Degree:** **Degree 11** with Ridge regularization ($\alpha = 2.0$) / **Degree 8** for standard OLS.
   - **Cross-Validation Score:** MSE = 0.2283, $R^2 = 0.9954$.

---

## File Structure

- `train.py`: Evaluates polynomial degrees using 10-fold cross-validation and prints performance metrics (MSE, $R^2$).
- `inference.py`: Trains the final models on the full training sets and generates the prediction CSV files.
- `requirements.txt`: Python package requirements.
- `BT2024119_pred_var1.csv`: Final test predictions for Phase 1.
- `BT2024119_pred_var2.csv`: Final test predictions for Phase 2.
- `report.tex`: LaTeX source code for the assignment report.
- `figures/`: Plots generated for degree validation, residual checks, and spatial heatmaps.
- `BT2024119_dataset/`: Training and test datasets assigned for Roll No. BT2024119.

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run model training & degree evaluation:**
   ```bash
   python train.py
   ```

3. **Generate test predictions:**
   ```bash
   python inference.py
   ```

The script will generate `BT2024119_pred_var1.csv` and `BT2024119_pred_var2.csv` formatted with a single `y` header and 1000 prediction rows.
