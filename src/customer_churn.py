"""
Telco Customer Churn Prediction

Amaç:
    IBM Telco Customer Churn veri setini kullanarak
    müşterilerin aboneliklerini iptal edip etmeyeceğini
    makine öğrenmesi ile tahmin etmek.

Kullanılan modeller:
    - Logistic Regression
    - KNN
    - Decision Tree
    - SVM

Değerlendirme:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix
    - Cross Validation
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("results", exist_ok=True)

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)


# 1. VERİ YÜKLEME VE İLK İNCELEME

df = pd.read_csv("data/Telco-Customer-Churn.csv")
print("Veri seti başarıyla yüklendi.")

print("\nİlk 5 Satır")
print(df.head())
print("\nSon 5 Satır")
print(df.tail())
print("\nVeri Seti Boyutu")
print(df.shape)
print("\nSütunlar")
print(df.columns.tolist())
print("\nVeri Tipleri")
print(df.dtypes)
print("\nDataFrame Info")
df.info()
print("\nİstatistiksel Özet")
print(df.describe())


# 2. VERİ TEMİZLEME VE ÖN İŞLEME

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nEksik Değerler")
print(df.isnull().sum())
print("\nTekrarlayan Satır Sayısı")
print(df.duplicated().sum())

if df.duplicated().sum() > 0:
    df = df.drop_duplicates()

df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


# 3. FEATURE ENGINEERING

df["AverageMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)


# 4. TRAIN / VALIDATION / TEST SPLIT

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train, Validation, Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print("\nVeri Seti Boyutları")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# 5. EXPLORATORY DATA ANALYSIS

df_train = X_train.copy()
df_train["Churn"] = y_train

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS (SADECE TRAIN SETI ÜZERINDE)")
print("=" * 60)

print("\nChurn Dağılımı")
print(df_train["Churn"].value_counts())
print("\nChurn Oranları")
print(df_train["Churn"].value_counts(normalize=True))

plt.figure(figsize=(7, 5))
sns.countplot(data=df_train, x="Churn")
plt.title("Customer Churn Distribution (Train Data)")
plt.xlabel("Churn")
plt.ylabel("Customer Count")
plt.tight_layout()
plt.savefig("results/churn_distribution.png")
plt.close()

print("\nChurn Oranı: Contract")
contract_churn = df_train.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
print(contract_churn)

plt.figure(figsize=(8, 5))
sns.barplot(x=contract_churn.index, y=contract_churn.values)
plt.title("Churn Rate by Contract Type (Train Data)")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/churn_by_contract.png")
plt.close()

print("\nChurn Oranı: Internet Service")
internet_churn = df_train.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)
print(internet_churn)

plt.figure(figsize=(8, 5))
sns.barplot(x=internet_churn.index, y=internet_churn.values)
plt.title("Churn Rate by Internet Service (Train Data)")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/churn_by_internet_service.png")
plt.close()

print("\nChurn Oranı: Tech Support")
support_churn = df_train.groupby("TechSupport")["Churn"].mean().sort_values(ascending=False)
print(support_churn)

plt.figure(figsize=(8, 5))
sns.barplot(x=support_churn.index, y=support_churn.values)
plt.title("Churn Rate by Tech Support (Train Data)")
plt.xlabel("Tech Support")
plt.ylabel("Churn Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/churn_by_tech_support.png")
plt.close()

print("\nChurn Oranı: Payment Method")
payment_churn = df_train.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False)
print(payment_churn)

plt.figure(figsize=(10, 5))
sns.barplot(x=payment_churn.index, y=payment_churn.values)
plt.title("Churn Rate by Payment Method (Train Data)")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate")
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/churn_by_payment_method.png")
plt.close()

print("\nTenure İstatistikleri")
print(df_train.groupby("Churn")["tenure"].describe())

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_train, x="Churn", y="tenure")
plt.title("Tenure Distribution by Churn (Train Data)")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.savefig("results/tenure_by_churn.png")
plt.close()

print("\nMonthly Charges İstatistikleri")
print(df_train.groupby("Churn")["MonthlyCharges"].describe())

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_train, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges Distribution by Churn (Train Data)")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("results/monthly_charges_by_churn.png")
plt.close()

print("\nTotal Charges İstatistikleri")
print(df_train.groupby("Churn")["TotalCharges"].describe())

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_train, x="Churn", y="TotalCharges")
plt.title("Total Charges Distribution by Churn (Train Data)")
plt.xlabel("Churn")
plt.ylabel("Total Charges")
plt.tight_layout()
plt.savefig("results/total_charges_by_churn.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_train, x="Contract", y="MonthlyCharges", hue="Churn")
plt.title("Monthly Charges by Contract and Churn (Train Data)")
plt.xlabel("Contract")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("results/contract_charges_churn.png")
plt.close()

print("\nChurn Oranı: Online Security")
security_churn = df_train.groupby("OnlineSecurity")["Churn"].mean().sort_values(ascending=False)
print(security_churn)

plt.figure(figsize=(8, 5))
sns.barplot(x=security_churn.index, y=security_churn.values)
plt.title("Churn Rate by Online Security (Train Data)")
plt.xlabel("Online Security")
plt.ylabel("Churn Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/churn_by_online_security.png")
plt.close()

print("\nChurn Oranı: Partner")
partner_churn = df_train.groupby("Partner")["Churn"].mean().sort_values(ascending=False)
print(partner_churn)

plt.figure(figsize=(7, 5))
sns.barplot(x=partner_churn.index, y=partner_churn.values)
plt.title("Churn Rate by Partner Status (Train Data)")
plt.xlabel("Partner")
plt.ylabel("Churn Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/churn_by_partner.png")
plt.close()

print("\nChurn Oranı: Dependents")
dependents_churn = df_train.groupby("Dependents")["Churn"].mean().sort_values(ascending=False)
print(dependents_churn)

plt.figure(figsize=(7, 5))
sns.barplot(x=dependents_churn.index, y=dependents_churn.values)
plt.title("Churn Rate by Dependents (Train Data)")
plt.xlabel("Dependents")
plt.ylabel("Churn Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/churn_by_dependents.png")
plt.close()

print("\nNumerical Değişkenlerin Korelasyonu")
numeric_df = df_train.select_dtypes(include=["int64", "float64"])
print(numeric_df.corr())

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f")
plt.title("Numerical Feature Correlation Matrix (Train Data)")
plt.tight_layout()
plt.savefig("results/correlation_matrix.png")
plt.close()

print("EDA TAMAMLANDI")

# 6. PREPROCESSING PIPELINE

numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "AverageMonthlySpend"]
categorical_features = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
    "PaperlessBilling", "PaymentMethod"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# 7. MODEL TRAINING

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

knn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsClassifier(n_neighbors=5))
])

tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(max_depth=5, random_state=42))
])

svm_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", SVC(kernel="rbf", random_state=42))
])

models = {
    "Logistic Regression": logistic_model,
    "KNN": knn_model,
    "Decision Tree": tree_model,
    "SVM": svm_model
}

validation_results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    
    validation_results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_val, y_val_pred),
        "Precision": precision_score(y_val, y_val_pred),
        "Recall": recall_score(y_val, y_val_pred),
        "F1 Score": f1_score(y_val, y_val_pred)
    })

validation_results_df = pd.DataFrame(validation_results)
print("\nValidation Sonuçları")
print(validation_results_df)


# 8. CROSS VALIDATION

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
scoring = ["accuracy", "precision", "recall", "f1"]

for model_name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    
    cv_results.append({
        "Model": model_name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1 Score": scores["test_f1"].mean()
    })

cv_results_df = pd.DataFrame(cv_results)
print("\nCross Validation Sonuçları")
print(cv_results_df)

# Grafik: Model Performans Karşılaştırması
cv_results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(kind="bar", figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("results/model_comparison.png")
plt.close()


# 9. FINAL MODEL AND TEST

best_model_name = cv_results_df.loc[cv_results_df["F1 Score"].idxmax(), "Model"]
print("\nEn İyi Model:", best_model_name)

best_model = models[best_model_name]

# Final eğitimi için train ve val setlerini birleştirme
X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])
best_model.fit(X_train_final, y_train_final)
y_test_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\nFinal Test Results")
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1 Score:", test_f1)

print("\nClassification Report")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# Final Sonuçların Özeti
final_results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [test_accuracy, test_precision, test_recall, test_f1]
})

print("\nFinal Model Results")
print(final_results)