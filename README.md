# Credit Card Default Prediction

This project aims to predict credit card default probabilities using various supervised machine learning algorithms. It explores the factors influencing credit card defaults and develops a predictive model to assist financial institutions in mitigating risks.

---

## **Overview**

Credit card defaults are a critical issue faced by financial institutions. This project analyzes data from Taiwanese banks collected between April 2005 and September 2005 to build a machine learning model that predicts whether a client will default on their payment in the next month. Key techniques such as data preprocessing, resampling, and model evaluation are applied to achieve the best results.

---

## **Dataset**

The dataset contains **30,000 instances** of credit card clients' demographic information, historical transactions, and repayment status. It includes:

- **Demographic Information**:
  - `LIMIT_BAL`: Credit limit.
  - `SEX`: Gender (1 = Male, 2 = Female).
  - `EDUCATION`: Education level.
  - `MARRIAGE`: Marital status.
  - `AGE`: Age in years.

- **Repayment History**:
  - `PAY_0` to `PAY_6`: Repayment status for the last 6 months.

- **Billing and Payment History**:
  - `BILL_AMT1` to `BILL_AMT6`: Bill amounts for the last 6 months.
  - `PAY_AMT1` to `PAY_AMT6`: Payment amounts for the last 6 months.

- **Target Variable**:
  - `default.payment.next.month`: Default status (1 = Default, 0 = Non-default).

---

## **Steps and Methods**

### **1. Exploratory Data Analysis (EDA)**
- Distribution analysis of the target variable revealed a **class imbalance** (78% non-defaults vs. 22% defaults).
- Continuous and categorical features were explored using visualizations such as:
  - Kernel Density Estimation (KDE).
  - QQ-plots for normality checks.
  - Correlation heatmaps.

### **2. Data Preprocessing**
- **Handling Missing and Anomalous Values**:
  - Invalid values in `MARRIAGE` and `EDUCATION` were remapped to "Others."
  - Outliers in `PAY` features were adjusted.
- **Feature Scaling**:
  - Min-Max Normalization.
  - Standardization (Z-score scaling).
- **One-Hot Encoding**:
  - Transformed categorical features like `SEX`, `MARRIAGE`, and `EDUCATION` into binary dummy variables.

### **3. Class Imbalance Handling**
- Resampling methods:
  - **SMOTE (Synthetic Minority Oversampling Technique)**.
  - **Cluster Centroid Undersampling**.

### **4. Machine Learning Models**
Implemented and compared the following algorithms:
- **Logistic Regression**.
- **Decision Tree**.
- **Random Forest**.
- **Support Vector Machine (SVM)**:
  - Hard Margin.
  - Soft Margin.
  - Kernel SVM.
- **Deep Learning**:
  - Multi-layer Neural Networks.

### **5. Evaluation Metrics**
- Accuracy, Precision, Recall, and F1-Score.
- Confusion Matrices and Precision-Recall Curves.

---

## **Results**

- **Logistic Regression**:
  - Accuracy: 81.37%
  - F1-Score: 36% (Default class).

- **Random Forest**:
  - Best performance with PCA and SMOTE.
  - Significant importance of repayment status, age, and bill amount.

- **Support Vector Machine**:
  - Hard margin and kernel trick improved separability.

- **Deep Learning**:
  - Training Accuracy: 77.8%.
  - Test Accuracy: 78.3%.
  - Low F1-Score indicates overfitting or insufficient feature learning.

---

## **Conclusion**

- Logistic Regression and Random Forest performed comparably in accuracy but had differing strengths:
  - Logistic Regression is better for minimizing false positives.
  - Random Forest provides better feature importance insights.
- Handling class imbalance using SMOTE and Cluster Centroids improved model performance.
- Deep learning did not perform well due to potential issues with hyperparameter tuning and dataset limitations.

---

## **Future Work**

- Experiment with advanced models such as **Gradient Boosting Machines (GBM)** or **XGBoost**.
- Explore anomaly detection methods like **Local Outlier Factor** or **Isolation Forest**.
- Incorporate more recent datasets with updated client behavior patterns.

---

## **References**

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

---

## **How to Use**

1. **Setup**:
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run the Notebook**:
   - Open `Group_8_Models.ipynb` in Jupyter Notebook or any compatible environment.

3. **Explore Results**:
   - Visualize EDA insights and model performance metrics in the notebook outputs.

---

This project provides valuable insights into credit risk modeling and demonstrates the application of machine learning techniques for predictive analytics.
