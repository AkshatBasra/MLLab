import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

from sklearn.decomposition import PCA

df = pd.read_csv("heart.csv")

print("First 5 Rows:")
print(df.head())

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

categorical_cols = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope"
]

numerical_cols = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak"
]

from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop='first'), categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SVM": SVC(kernel='rbf'),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

print("\n==============================")
print("WITHOUT PCA")
print("==============================")

results = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"{name} Accuracy: {acc:.4f}")

best_model = max(results, key=results.get)
print("\nBest Model Without PCA:", best_model)

print("\n==============================")
print("WITH PCA")
print("==============================")

pca_results = {}

for name, model in models.items():

    pipeline_pca = Pipeline([
        ("preprocessing", preprocessor),
        ("pca", PCA(n_components=0.95)),
        ("classifier", model)
    ])

    pipeline_pca.fit(X_train, y_train)

    y_pred = pipeline_pca.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pca_results[name] = acc

    print(f"{name} Accuracy with PCA: {acc:.4f}")

best_model_pca = max(pca_results, key=pca_results.get)
print("\nBest Model With PCA:", best_model_pca)

print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

for name in models.keys():
    print(f"{name}")
    print(f"Without PCA : {results[name]:.4f}")
    print(f"With PCA    : {pca_results[name]:.4f}")
    print("---------------------------")