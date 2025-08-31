pip install ucimlrepo

from ucimlrepo import fetch_ucirepo
# fetch dataset
heart_disease = fetch_ucirepo(id=45)
# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# metadata
heart_disease.metadata

# variable information
heart_disease.variables

x

y

y.value_counts()

# EDA for Heart Disease dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Combine into one DataFrame
data = pd.concat([X, y], axis=1)
data = data.dropna()  # drop NaN rows

print("Dataset shape:", data.shape)

print("\nData Types:\n", data.dtypes)

print("\nMissing Values:\n", data.isnull().sum())

print("\nTarget Distribution:\n", data[y.columns[0]].value_counts())

# Summary statistics
print("\nSummary Statistics:\n", data.describe())

# 1. Target distribution plot
plt.figure(figsize=(6,4))
sns.countplot(x=y.columns[0], data=data, palette="Set2")
plt.title("Distribution of Heart Disease (Target)")
plt.xlabel("Target (0 = No Disease, 1+ = Disease)")
plt.ylabel("Count")
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 3. Distribution of some key features
features_to_plot = ["age", "trestbps", "chol", "thalach"]
for feature in features_to_plot:
    plt.figure(figsize=(6,4))
    sns.histplot(data[feature], kde=True, bins=30, color="skyblue")
    plt.title(f"Distribution of {feature}")
    plt.show()

# 4. Feature vs Target (Boxplots)
for feature in features_to_plot:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=y.columns[0], y=feature, data=data, palette="Set3")
    plt.title(f"{feature} vs Heart Disease")
    plt.xlabel("Target")
    plt.ylabel(feature)
    plt.show()

# Import libraries
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Step 1: Load dataset
heart_disease = fetch_ucirepo(id=45)   # Cleveland Heart Disease dataset
X = heart_disease.data.features
y = heart_disease.data.targets
print("Dataset Shape (before cleaning):", X.shape)

# Step 2: Remove NaN values
data = pd.concat([X, y], axis=1)
data = data.dropna()   # removes rows with NaN
X = data.drop(columns=y.columns)  # features
y = data[y.columns[0]]            # target

print("Dataset Shape (after cleaning):", X.shape)

# Step 3: Preprocessing (scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest1": RandomForestClassifier(n_estimators=100, random_state=42),
    "Random Forest2": RandomForestClassifier(n_estimators=50, random_state=42),
    "Random Forest3": RandomForestClassifier(n_estimators=150, random_state=42),
    "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=42)
}

# Step 6: Train and Evaluate
results = []
for name, model in models.items():
    print(name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    # auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
    results.append([name, acc, prec, rec, f1])
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))

# Step 7: Results Summary
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"])
print("\nModel Performance Comparison:")
print(results_df)
