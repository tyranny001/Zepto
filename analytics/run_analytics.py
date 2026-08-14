"""Titanic analytics Part A + Part B (executable script mirroring notebook)."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

DATA_PATH = Path(__file__).resolve().parent / "titanic.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PIPELINE_PATH = Path(__file__).resolve().parent / "survival_pipeline.joblib"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def part_a(df: pd.DataFrame) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("=== Part A: Profile ===")
    print(df.info())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())

    clean = df.drop_duplicates().copy()
    clean["age"] = clean["age"].fillna(clean["age"].median())
    clean["embarked"] = clean["embarked"].fillna(clean["embarked"].mode()[0])
    clean["embark_town"] = clean["embark_town"].fillna(clean["embark_town"].mode()[0])
    clean["deck"] = clean["deck"].fillna("Unknown")
    clean["survived"] = clean["survived"].astype(int)

    numeric = clean.select_dtypes(include=[np.number])
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=120)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.countplot(data=clean, x="survived", ax=axes[0, 0])
    axes[0, 0].set_title("Overall survival count")
    sns.barplot(data=clean, x="class", y="survived", estimator="mean", ax=axes[0, 1])
    axes[0, 1].set_title("Survival rate by passenger class")
    sns.barplot(data=clean, x="sex", y="survived", estimator="mean", ax=axes[1, 0])
    axes[1, 0].set_title("Survival rate by sex")
    sns.histplot(clean, x="age", hue="survived", kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Age distribution by survival")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_charts.png", dpi=120)
    plt.close()

    scaler = StandardScaler()
    scaled_age = scaler.fit_transform(clean[["age"]])
    print(
        f"\nStandardization check (age): mean={scaled_age.mean():.4f}, std={scaled_age.std():.4f}"
    )

    clean.to_csv(Path(__file__).resolve().parent / "titanic_clean.csv", index=False)
    return clean


def build_preprocessor(feature_cols: list[str], cat_cols: list[str], num_cols: list[str]):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def part_b(clean: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    cat_cols = ["sex", "embarked"]
    num_cols = ["pclass", "age", "sibsp", "parch", "fare"]
    X = clean[feature_cols]
    y = clean["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(feature_cols, cat_cols, num_cols)

    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = []
    best_pipe = None
    best_f1 = -1.0

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds),
                "recall": recall_score(y_test, preds),
                "f1": f1_score(y_test, preds),
                "roc_auc": roc_auc_score(y_test, proba),
            }
        )
        if results[-1]["f1"] > best_f1:
            best_f1 = results[-1]["f1"]
            best_pipe = pipe

        if name == "DecisionTree":
            plt.figure(figsize=(20, 10))
            plot_tree(
                pipe.named_steps["clf"],
                feature_names=pipe.named_steps["prep"].get_feature_names_out(),
                class_names=["died", "survived"],
                filled=True,
                max_depth=3,
            )
            plt.title("Decision Tree (top levels)")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "decision_tree.png", dpi=100)
            plt.close()

    metrics_df = pd.DataFrame(results)
    print("\n=== Classifier metrics ===")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(OUTPUT_DIR / "classifier_metrics.csv", index=False)

    # Imbalance comparison: class_weight vs none
    rf_balanced = Pipeline([
        ("prep", build_preprocessor(feature_cols, cat_cols, num_cols)),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
    ])
    rf_default = Pipeline([
        ("prep", build_preprocessor(feature_cols, cat_cols, num_cols)),
        ("clf", RandomForestClassifier(random_state=42)),
    ])
    rf_balanced.fit(X_train, y_train)
    rf_default.fit(X_train, y_train)
    print("\n=== Imbalance handling (RF F1) ===")
    print("balanced:", f1_score(y_test, rf_balanced.predict(X_test)))
    print("default:", f1_score(y_test, rf_default.predict(X_test)))

    # GridSearchCV + OOB
    rf_pipe = Pipeline([
        ("prep", build_preprocessor(feature_cols, cat_cols, num_cols)),
        ("clf", RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)),
    ])
    param_grid = {"clf__n_estimators": [50, 100], "clf__max_depth": [None, 5, 8]}
    grid = GridSearchCV(rf_pipe, param_grid, cv=StratifiedKFold(3), scoring="f1")
    grid.fit(X_train, y_train)
    print("\n=== GridSearchCV best params ===", grid.best_params_)
    print("OOB score:", grid.best_estimator_.named_steps["clf"].oob_score_)

    # Regression side-task: predict fare
    reg_X = clean[["pclass", "age", "sibsp", "parch", "survived"]]
    reg_y = clean["fare"]
    reg_pipe = Pipeline([
        ("prep", build_preprocessor(["pclass", "age", "sibsp", "parch", "survived"], [], ["pclass", "age", "sibsp", "parch", "survived"])),
        ("reg", LinearRegression()),
    ])
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(reg_X, reg_y, test_size=0.2, random_state=42)
    reg_pipe.fit(Xr_train, yr_train)
    mse = mean_squared_error(yr_test, reg_pipe.predict(Xr_test))
    rmse = np.sqrt(mse)
    print(f"\n=== Regression fare RMSE: {rmse:.2f} ===")

    # Save and reload pipeline
    assert best_pipe is not None
    joblib.dump(best_pipe, PIPELINE_PATH)
    reloaded = joblib.load(PIPELINE_PATH)
    orig_preds = best_pipe.predict(X_test)
    reload_preds = reloaded.predict(X_test)
    assert np.array_equal(orig_preds, reload_preds)
    print(f"Pipeline saved to {PIPELINE_PATH.name}; reload predictions match.")


def main() -> None:
    df = load_data()
    clean = part_a(df)
    part_b(clean)
    print("\nAnalytics complete.")


if __name__ == "__main__":
    main()
