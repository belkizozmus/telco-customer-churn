# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to cancel their subscription.

## Project Overview

Customer churn is an important business problem for telecommunication companies.
The goal of this project is to analyze customer information and build machine learning models that can predict whether a customer will leave the company.
The IBM Telco Customer Churn dataset is used in this project.

## Objectives

*   Explore and understand the Telco Customer Churn dataset
*   Handle missing and inconsistent values
*   Perform Exploratory Data Analysis (EDA)
*   Create and evaluate useful features
*   Encode categorical variables
*   Scale numerical variables
*   Train multiple classification models
*   Compare model performance
*   Use cross-validation
*   Select the best-performing model
*   Evaluate the final model on unseen test data

## Dataset

The dataset contains customer information such as:
*   Customer demographics
*   Tenure
*   Contract type
*   Internet service
*   Online services
*   Payment method
*   Monthly charges
*   Total charges
*   Churn status

The original dataset contains 7,043 customers and 21 columns. 

The target variable is:
*   `Churn = 1` → Customer churned
*   `Churn = 0` → Customer did not churn

## Project Structure

```text
telco-customer-churn/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── src/
│   └── customer_churn.py
│
├── results/
│   ├── churn_distribution.png
│   ├── churn_by_contract.png
│   ├── churn_by_internet_service.png
│   ├── churn_by_tech_support.png
│   ├── churn_by_payment_method.png
│   ├── tenure_by_churn.png
│   ├── monthly_charges_by_churn.png
│   ├── total_charges_by_churn.png
│   ├── contract_charges_churn.png
│   ├── churn_by_online_security.png
│   ├── churn_by_partner.png
│   ├── churn_by_dependents.png
│   ├── correlation_matrix.png
│   ├── model_comparison.png
│   └── confusion_matrix.png
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Data Preprocessing

The following preprocessing steps were applied:
*   Loaded the dataset using Pandas.
*   Converted `TotalCharges` from string to numeric.
*   Checked missing values and duplicated rows.
*   Removed the `customerID` column because it is only an identifier and does not provide useful predictive information.
*   Converted the `Churn` target variable: Yes → 1, No → 0.
*   Created the `AverageMonthlySpend` feature for exploratory analysis.
*   Split the dataset into training, validation, and test sets using stratification.
*   Applied median imputation and standardization to numerical features.
*   Applied most-frequent imputation and one-hot encoding to categorical features.

## Train / Validation / Test Split

The dataset was divided into three parts using stratified splitting to preserve the churn class distribution:
*   **Training set:** 70% (4,930 samples)
*   **Validation set:** 15% (1,056 samples)
*   **Test set:** 15% (1,057 samples)

## Exploratory Data Analysis

EDA was performed only on the training dataset to avoid using validation or test data during exploratory analysis.

### Churn Distribution
The training dataset contains approximately:
*   **73.5%** customers who did not churn
*   **26.5%** customers who churned
*(This shows that the target variable is somewhat imbalanced.)*

### Key Features and Churn Rates

**Contract Type**
Customers with month-to-month contracts have a substantially higher churn rate.
| Contract Type | Churn Rate |
| :--- | :--- |
| Month-to-month | 42.86% |
| One year | 10.93% |
| Two year | 3.02% |

**Internet Service**
Fiber optic customers have a higher churn rate in this dataset.
| Internet Service | Churn Rate |
| :--- | :--- |
| Fiber optic | 41.85% |
| DSL | 19.17% |
| No internet service | 6.93% |

**Tech Support**
Customers without technical support have a considerably higher churn rate.
| Tech Support | Churn Rate |
| :--- | :--- |
| No | 41.93% |
| Yes | 15.37% |
| No internet service | 6.93% |

**Payment Method**
Customers using electronic check have the highest churn rate among the payment methods.
| Payment Method | Churn Rate |
| :--- | :--- |
| Electronic check | 45.65% |
| Mailed check | 19.50% |
| Bank transfer (automatic) | 16.30% |
| Credit card (automatic) | 14.71% |

**Tenure & Charges**
Customers who churn generally have shorter subscription periods and higher monthly charges.
| Feature | No Churn (Avg) | Churn (Avg) |
| :--- | :--- | :--- |
| **Tenure** | 37.56 months | 18.38 months |
| **Monthly Charges** | 61.32 | 75.00 |

### Feature Correlation
*   **Tenure - Churn correlation (-0.34):** Suggests that customers with longer tenure are generally less likely to churn.
*   **MonthlyCharges - Churn correlation (0.20):** Indicates that higher monthly charges are associated with a higher probability of churn.
*   **TotalCharges - Tenure correlation (0.83):** Expected, as total charges increase as the customer's tenure increases.

## Machine Learning Models

Four classification algorithms were evaluated. All preprocessing steps were implemented using Scikit-learn pipelines.

1.  **Logistic Regression:** Used as a linear classification model. It provides a strong and interpretable baseline for binary classification problems.
2.  **K-Nearest Neighbors (KNN):** Classifies customers based on the characteristics of their nearest observations (`n_neighbors = 5`).
3.  **Decision Tree:** Used to model nonlinear relationships between customer characteristics and churn (`max_depth = 5` to reduce overfitting).
4.  **Support Vector Machine (SVM):** An SVM with an RBF kernel was used to model nonlinear decision boundaries.

## Model Evaluation

The following metrics were used: Accuracy, Precision, Recall, F1 Score, and Confusion Matrix. F1 Score was selected as the main criterion for choosing the final model because it provides a balance between Precision and Recall.

### Cross-Validation Results
A 5-fold Stratified Cross-Validation strategy was used.

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.803** | 0.654 | **0.547** | **0.595** |
| KNN | 0.763 | 0.556 | 0.532 | 0.544 |
| Decision Tree | 0.787 | 0.606 | 0.563 | 0.583 |
| SVM | 0.804 | **0.678** | 0.500 | 0.575 |

According to the F1 Score, **Logistic Regression** achieved the best overall balance between Precision and Recall and was selected as the final model.

## Final Model

After selecting Logistic Regression, the model was retrained using combined Training + Validation Data and evaluated only once on the previously unseen test dataset.

### Final Test Results

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.809 |
| **Precision** | 0.679 |
| **Recall** | 0.534 |
| **F1 Score** | 0.598 |

**Interpretation:**
*   The final model achieved approximately 80.9% accuracy.
*   **Precision (67.9%):** Among the customers predicted as churners, about 68% actually churned.
*   **Recall (53.4%):** The model identified about 53% of the customers who actually churned.
*   **F1 Score (59.8%):** Provides a balanced view of the model's performance.

### Classification Report
```text
              precision    recall  f1-score   support

           0       0.84      0.91      0.87       776
           1       0.68      0.53      0.60       281

    accuracy                           0.81      1057
   macro avg       0.76      0.72      0.74      1057
weighted avg       0.80      0.81      0.80      1057
```

### Confusion Matrix
```text
[[705  71]
 [131 150]]
```
*   **705** customers were correctly classified as non-churn.
*   **71** non-churn customers were incorrectly classified as churn.
*   **131** churn customers were incorrectly classified as non-churn. *(Main weakness)*
*   **150** churn customers were correctly classified as churn.

## Results Visualization

The project generates several visualizations in the `results/` directory, including:
*   Churn distribution
*   Churn rate by contract, internet service, technical support, payment method, online security, partner status, and dependents.
*   Tenure, monthly charges, and total charges by churn.
*   Feature correlation matrix.
*   Model performance comparison and Final confusion matrix.

## Technologies
*   Python (Pandas, NumPy)
*   Scikit-learn
*   Matplotlib, Seaborn
*   Git, GitHub

## Installation & Running the Project

**1. Clone the repository:**
```bash
git clone [https://github.com/belkizozmus/telco-customer-churn.git](https://github.com/belkizozmus/telco-customer-churn.git)
cd telco-customer-churn
```

**2. Create and activate a virtual environment (Windows):**
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the main Python script:**
```bash
python src/customer_churn.py
```
*(After execution, the generated charts will be saved in the `results/` directory.)*

## Conclusion

This project demonstrates a complete machine learning workflow for customer churn prediction:

Data Loading → Data Inspection → Data Cleaning → Feature Engineering → Train / Validation / Test Split → Exploratory Data Analysis → Preprocessing → Model Training → Validation → Cross-Validation → Model Selection → Final Training → Test Evaluation → Confusion Matrix.

Among the evaluated models, Logistic Regression achieved the highest F1 Score (59.8%) on the test dataset. Although the model provides a reasonable baseline, the churn recall of approximately 53.4% indicates that there is still room for improvement, particularly in identifying customers who are likely to leave.

## Dataset Source
The project uses the IBM Telco Customer Churn dataset. The dataset is included in the `data/` directory for this project.
