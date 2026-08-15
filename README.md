# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to cancel their subscription.

## Project Overview

Customer churn is an important business problem for telecommunication companies.

The goal of this project is to analyze customer information and build machine learning models that can predict whether a customer will leave the company.

The IBM Telco Customer Churn dataset is used in this project.

## Objectives

- Explore and understand the Telco Customer Churn dataset
- Handle missing and inconsistent values
- Perform Exploratory Data Analysis (EDA)
- Create and evaluate useful features
- Encode categorical variables
- Scale numerical variables
- Train multiple classification models
- Compare model performance
- Use cross-validation
- Select the best-performing model
- Evaluate the final model on unseen test data

## Dataset

The dataset contains customer information such as:

- Customer demographics
- Tenure
- Contract type
- Internet service
- Online services
- Payment method
- Monthly charges
- Total charges
- Churn status

The original dataset contains 7,043 customers and 21 columns.

The target variable is:

- `Churn = 1` → Customer churned
- `Churn = 0` → Customer did not churn

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
Data Preprocessing
The following preprocessing steps were applied:

Loaded the dataset using Pandas.
Converted TotalCharges from string to numeric.
Checked missing values.
Checked duplicated rows.
Removed the customerID column because it is only an identifier and does not provide useful predictive information.
Converted the Churn target variable:
Yes → 1
No → 0
Created the AverageMonthlySpend feature for exploratory analysis.
Split the dataset into training, validation, and test sets using stratification.
Applied median imputation and standardization to numerical features.
Applied most-frequent imputation and one-hot encoding to categorical features.
Train / Validation / Test Split

The dataset was divided into three parts:

Training set: 70%
Validation set: 15%
Test set: 15%

Stratified splitting was used to preserve the churn class distribution.

Training:   4930 samples
Validation: 1056 samples
Test:       1057 samples
Exploratory Data Analysis

EDA was performed only on the training dataset to avoid using validation or test data during exploratory analysis.

Churn Distribution
The training dataset contains approximately:

73.5% customers who did not churn
26.5% customers who churned

This shows that the target variable is somewhat imbalanced.

Contract Type

The churn rate differs significantly by contract type.

Contract Type	Churn Rate
Month-to-month	42.86%
One year	10.93%
Two year	3.02%

Customers with month-to-month contracts have a substantially higher churn rate.

Internet Service
Internet Service	Churn Rate
Fiber optic	41.85%
DSL	19.17%
No internet service	6.93%

Fiber optic customers have a higher churn rate in this dataset.

Tech Support
Tech Support	Churn Rate
No	41.93%
Yes	15.37%
No internet service	6.93%

Customers without technical support have a considerably higher churn rate.

Payment Method
Payment Method	Churn Rate
Electronic check	45.65%
Mailed check	19.50%
Bank transfer (automatic)	16.30%
Credit card (automatic)	14.71%

Customers using electronic check have the highest churn rate among the payment methods.

Tenure
Customers who churn generally have shorter subscription periods.

Average tenure:
Churn	Average Tenure
No Churn	37.56 months
Churn	18.38 months

This indicates that customer tenure is an important factor associated with churn.

Monthly Charges
Average monthly charges:
Churn	Average Monthly Charges
No Churn	61.32
Churn	75.00

Customers who churn generally have higher monthly charges.

Feature Correlation
The correlation analysis showed that tenure has a negative relationship with churn:
tenure - Churn correlation: -0.34
This suggests that customers with longer tenure are generally less likely to churn.

MonthlyCharges has a positive relationship with churn:
MonthlyCharges - Churn correlation: 0.20

This indicates that higher monthly charges are associated with a higher probability of churn.

TotalCharges is strongly correlated with tenure:
TotalCharges - tenure correlation: 0.83

This is expected because total charges generally increase as the customer's tenure increases.

Machine Learning Models
Four classification algorithms were evaluated:
Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Support Vector Machine (SVM)

All preprocessing steps were implemented using Scikit-learn pipelines.

Logistic Regression
Logistic Regression was used as a linear classification model.
It provides a strong and interpretable baseline for binary classification problems.

K-Nearest Neighbors
KNN classifies customers based on the characteristics of their nearest observations.

The model was configured with:
n_neighbors = 5
Decision Tree

A Decision Tree was used to model nonlinear relationships between customer characteristics and churn.

The tree depth was limited to:
max_depth = 5
to reduce the risk of overfitting.

Support Vector Machine
An SVM with an RBF kernel was used to model nonlinear decision boundaries.

Model Evaluation
The following metrics were used:
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
5-Fold Stratified Cross-Validation

F1 Score was selected as the main criterion for choosing the final model because it provides a balance between Precision and Recall.

Cross-Validation
A 5-fold Stratified Cross-Validation strategy was used.

StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
Cross-Validation Results
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	0.803	0.654	0.547	0.595
KNN	0.763	0.556	0.532	0.544
Decision Tree	0.787	0.606	0.563	0.583
SVM	0.804	0.678	0.500	0.575

According to the F1 Score, Logistic Regression achieved the best overall balance between Precision and Recall.
Therefore, Logistic Regression was selected as the final model.

Final Model
After selecting Logistic Regression, the training and validation datasets were combined.

The final model was then retrained using:
Training + Validation Data
The model was evaluated only once on the previously unseen test dataset.

Final Test Results
Metric	Score
Accuracy	0.809
Precision	0.679
Recall	0.534
F1 Score	0.598
Interpretation

The final model achieved approximately 80.9% accuracy.
The Precision score of approximately 67.9% means that among the customers predicted as churners, about 68% actually churned.
The Recall score of approximately 53.4% means that the model identified about 53% of the customers who actually churned.
The F1 Score was approximately 59.8%.

Classification Report
              precision    recall  f1-score   support


           0       0.84      0.91      0.87       776
           1       0.68      0.53      0.60       281


    accuracy                           0.81      1057
   macro avg       0.76      0.72      0.74      1057
weighted avg       0.80      0.81      0.80      1057
Confusion Matrix
[[705  71]
 [131 150]]

The confusion matrix can be interpreted as follows:
705 customers were correctly classified as non-churn.
71 non-churn customers were incorrectly classified as churn.
131 churn customers were incorrectly classified as non-churn.
150 churn customers were correctly classified as churn.

The main weakness of the model is that it misses 131 customers who actually churned.

Results Visualization
The project generates several visualizations in the results/ directory, including:
Churn distribution
Churn rate by contract
Churn rate by internet service
Churn rate by technical support
Churn rate by payment method
Tenure distribution by churn
Monthly charges by churn
Total charges by churn
Contract and monthly charges by churn
Churn rate by online security
Churn rate by partner status
Churn rate by dependents
Feature correlation matrix
Model performance comparison
Final confusion matrix
Technologies
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Git
GitHub
Installation

Clone the repository:
git clone https://github.com/YOUR_USERNAME/telco-customer-churn.git

Navigate to the project directory:
cd telco-customer-churn

Create a virtual environment:
Windows
python -m venv env

Activate the virtual environment:
.\env\Scripts\Activate.ps1

Install the required dependencies:
pip install -r requirements.txt
Running the Project

Run the main Python script:
python src/customer_churn.py

After execution, the generated charts will be saved in the:
results/
directory.

Future Improvements
Possible future improvements include:
Hyperparameter tuning
Class imbalance handling
Threshold optimization
Feature selection
Trying Random Forest
Trying Gradient Boosting
Improving churn recall
Using ROC-AUC
Using PR-AUC
Comparing additional classification algorithms
Conclusion

This project demonstrates a complete machine learning workflow for customer churn prediction.

The workflow includes:
Data Loading
      ↓
Data Inspection
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Train / Validation / Test Split
      ↓
Exploratory Data Analysis
      ↓
Preprocessing
      ↓
Model Training
      ↓
Validation
      ↓
Cross-Validation
      ↓
Model Selection
      ↓
Final Training
      ↓
Test Evaluation
      ↓
Confusion Matrix

Among the evaluated models, Logistic Regression achieved the highest F1 Score and was selected as the final model.
The final model achieved approximately 80.9% accuracy and 59.8% F1 Score on the test dataset.
Although the model provides a reasonable baseline, the churn recall of approximately 53.4% indicates that there is still room for improvement, particularly in identifying customers who are likely to leave.

Dataset Source
The project uses the IBM Telco Customer Churn dataset.
The dataset is included in the data/ directory for this project.
