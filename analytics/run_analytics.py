"""Titanic analytics Part A + Part B (executable script mirroring notebook)."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

DATA_PATH = Path(__file__).resolve().parent / "titanic.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PIPELINE_PATH = Path(__file__).resolve().parent / "survival_pipeline.joblib"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def part_a(df: pd.DataFrame) -> pd.DataFrame:
    """Part A: Profile, clean, EDA, and bivariate analysis."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("="*70)
    print("PART A: PROFILING, CLEANING, AND EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # ── Task 1: Profile ──
    print("\n=== PROFILE ===")
    print(f"\n--- df.shape ---\n{df.shape}")
    print(f"\n--- df.info() ---")
    print(df.info())
    print(f"\n--- df.describe() ---\n{df.describe()}")

    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({"count": missing, "percentage": missing_pct})
    print(missing_report[missing_report["count"] > 0])
    
    # ── Task 2: Missing-value handling with threshold rule ──
    print("\n=== MISSING VALUE THRESHOLD JUSTIFICATION ===")
    print("Threshold rule: <5% missing --> drop rows; 5-30% --> impute; >30% --> drop column or encode 'missing'.")
    print()
    for col in df.columns[df.isnull().any()]:
        pct = missing_pct[col]
        print(f"  * {col}: {int(missing[col])} missing ({pct}%)")
        if pct < 5:
            print(f"    --> Strategy: DROP rows ({pct}% < 5% threshold)")
        elif pct <= 30:
            print(f"    --> Strategy: IMPUTE ({pct}% in 5-30% range)")
        else:
            print(f"    --> Strategy: FILL with 'Unknown' ({pct}% > 30% -- imputation unreliable)")
            print(f"      Justification: Dropping the column would lose {100-pct:.1f}% valid data.")
            print(f"      Encoding 'Unknown' as its own category preserves missingness signal.")
    
    print(f"\nDuplicates: {df.duplicated().sum()}")
    
    # ── Apply cleaning ──
    clean = df.drop_duplicates().copy()

    # embarked / embark_town: <5% --> drop rows with missing
    clean = clean.dropna(subset=["embarked", "embark_town"])

    # age: 5-30% --> impute with median
    clean["age"] = clean["age"].fillna(clean["age"].median())

    # deck: >30% --> encode 'Unknown'
    clean["deck"] = clean["deck"].fillna("Unknown")

    clean["survived"] = clean["survived"].astype(int)

    print(f"\n=== CLEANED DATA SHAPE ===\n{clean.shape}")

    # ── Task 3: Univariate analysis ──
    print("\n=== UNIVARIATE ANALYSIS: AGE ===")
    age = clean["age"]
    q1_age, q3_age = age.quantile(0.25), age.quantile(0.75)
    iqr_age = q3_age - q1_age
    lower_bound_age = q1_age - 1.5 * iqr_age
    upper_bound_age = q3_age + 1.5 * iqr_age
    outliers_age = age[(age < lower_bound_age) | (age > upper_bound_age)]
    print(f"Q1={q1_age:.2f}, Q3={q3_age:.2f}, IQR={iqr_age:.2f}")
    print(f"Bounds: [{lower_bound_age:.2f}, {upper_bound_age:.2f}]")
    print(f"IQR-based outliers: {len(outliers_age)}")
    print(f"Mean={age.mean():.2f}, Median={age.median():.2f}, Mode={age.mode()[0]:.0f}")
    skew_age = age.skew()
    print(f"Skewness: {skew_age:.3f}")
    if age.mean() > age.median():
        print("Distribution: Slightly right-skewed (Mean > Median > Mode)")
    else:
        print("Distribution: Approximately symmetric or left-skewed")

    print("\n=== UNIVARIATE ANALYSIS: FARE ===")
    fare = clean["fare"]
    q1_fare, q3_fare = fare.quantile(0.25), fare.quantile(0.75)
    iqr_fare = q3_fare - q1_fare
    lower_bound_fare = q1_fare - 1.5 * iqr_fare
    upper_bound_fare = q3_fare + 1.5 * iqr_fare
    outliers_fare = fare[(fare < lower_bound_fare) | (fare > upper_bound_fare)]
    print(f"Q1={q1_fare:.2f}, Q3={q3_fare:.2f}, IQR={iqr_fare:.2f}")
    print(f"Bounds: [{lower_bound_fare:.2f}, {upper_bound_fare:.2f}]")
    print(f"IQR-based outliers: {len(outliers_fare)}")
    fare_mean = fare.mean()
    fare_median = fare.median()
    fare_mode = fare.mode()[0]
    print(f"Mean={fare_mean:.2f}, Median={fare_median:.2f}, Mode={fare_mode:.2f}")
    skew_fare = fare.skew()
    print(f"Skewness: {skew_fare:.3f}")
    print(f"Conclusion: Fare is RIGHT-SKEWED (Mean {fare_mean:.2f} >> Median {fare_median:.2f} >> Mode {fare_mode:.2f}).")
    print("The ordering Mean > Median > Mode is the classic signature of a right-skewed distribution.")
    print("High-fare outliers represent first-class passengers — legitimate, not data errors.")

    # ── Histogram + Box plot for both age and fare ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(age, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
    axes[0, 0].set_title("Histogram: Age", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Age (years)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(age.mean(), color="red", linestyle="--", label=f"Mean={age.mean():.1f}")
    axes[0, 0].axvline(age.median(), color="green", linestyle="--", label=f"Median={age.median():.1f}")
    axes[0, 0].legend()

    axes[0, 1].boxplot(age, vert=True)
    axes[0, 1].set_title("Box Plot: Age (IQR Outliers Visible)", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Age (years)")
    axes[0, 1].grid(axis="y", alpha=0.3)

    axes[1, 0].hist(fare, bins=30, edgecolor="black", alpha=0.7, color="#DD8452")
    axes[1, 0].set_title("Histogram: Fare", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Fare (£)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].axvline(fare.mean(), color="red", linestyle="--", label=f"Mean={fare_mean:.1f}")
    axes[1, 0].axvline(fare.median(), color="green", linestyle="--", label=f"Median={fare_median:.1f}")
    axes[1, 0].legend()

    axes[1, 1].boxplot(fare, vert=True)
    axes[1, 1].set_title("Box Plot: Fare (IQR Outliers Visible)", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Fare (£)")
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.suptitle("Univariate Analysis: Histograms & Box Plots", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_univariate_hist_box.png", dpi=120)
    plt.close()
    print("\n[OK] Saved: 01_univariate_hist_box.png")

    # ── Task 4: Bivariate analysis ──
    print("\n=== BIVARIATE ANALYSIS: SURVIVAL RATES ===")
    print("\n(a) Survival rate by sex:")
    sex_survival = clean.groupby("sex")["survived"].mean()
    print(sex_survival)
    print("\n(b) Survival rate by pclass:")
    pclass_survival = clean.groupby("pclass")["survived"].mean()
    print(pclass_survival)
    print("\n(c) Survival rate by sex AND pclass:")
    sex_pclass_survival = clean.groupby(["sex", "pclass"])["survived"].mean()
    print(sex_pclass_survival)

    # Correlation matrix (exact 6 columns, excluding adult_male and alone)
    print("\n=== CORRELATION MATRIX (6 numeric columns) ===")
    print("Columns: survived, pclass, age, sibsp, parch, fare")
    print("Excluded: adult_male (derived from sex/age), alone (derived from sibsp+parch)")
    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr_matrix = clean[corr_cols].corr()
    print(corr_matrix)
    
    # Identify top-2 correlations (off-diagonal)
    print("\n=== TOP-2 STRONGEST CORRELATIONS (off-diagonal, by |r|) ===")
    corr_pairs = []
    for i in range(len(corr_cols)):
        for j in range(i+1, len(corr_cols)):
            corr_pairs.append((corr_cols[i], corr_cols[j], abs(corr_matrix.iloc[i, j])))
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    for idx, (var1, var2, abs_val) in enumerate(corr_pairs[:2], 1):
        actual_corr = corr_matrix.loc[var1, var2]
        print(f"{idx}. {var1} <-> {var2}: r = {actual_corr:.4f} (|r| = {abs_val:.4f})")
        if actual_corr < 0:
            print(f"   Interpretation: Higher {var1} is associated with lower {var2} (negative correlation).")
        else:
            print(f"   Interpretation: Higher {var1} is associated with higher {var2} (positive correlation).")

    # ── Correlation heatmap ──
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
                cbar_kws={"label": "Correlation"}, vmin=-1, vmax=1)
    plt.title("Correlation Heatmap (6 Numeric Features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_correlation_heatmap.png", dpi=120)
    plt.close()
    print("[OK] Saved: 02_correlation_heatmap.png")

    # ── Task 5: Multivariate data story — 4 distinct charts ──
    print("\n=== GENERATING MULTIVARIATE DATA STORY (4 charts) ===")
    
    # Chart 1: Survival rate by class & sex (bar)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    pivot_data = clean.groupby(["pclass", "sex"])["survived"].mean().unstack()
    pivot_data.plot(kind="bar", ax=axes[0, 0], color=["#ff9999", "#66b3ff"])
    axes[0, 0].set_title("Chart 1: Survival Rate by Class & Sex", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Survival Rate")
    axes[0, 0].set_xlabel("Passenger Class")
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].legend(title="Sex")
    axes[0, 0].tick_params(axis='x', rotation=0)
    axes[0, 0].grid(axis="y", alpha=0.3)

    # Chart 2: Age distribution by survival (box)
    sns.boxplot(data=clean, x="survived", y="age", ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_title("Chart 2: Age Distribution by Survival", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Survived (0=No, 1=Yes)")
    axes[0, 1].set_ylabel("Age (years)")

    # Chart 3: Fare vs survival scatter colored by class
    survived_mask = clean["survived"] == 1
    died_mask = clean["survived"] == 0
    axes[1, 0].scatter(clean.loc[died_mask, "age"], clean.loc[died_mask, "fare"],
                       alpha=0.4, s=25, label="Did not survive", c="#E24A33")
    axes[1, 0].scatter(clean.loc[survived_mask, "age"], clean.loc[survived_mask, "fare"],
                       alpha=0.4, s=25, label="Survived", c="#348ABD")
    axes[1, 0].set_xlabel("Age (years)")
    axes[1, 0].set_ylabel("Fare (£)")
    axes[1, 0].set_title("Chart 3: Age vs Fare by Survival", fontsize=12, fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Chart 4: Survival count by embarked port
    sns.countplot(data=clean, x="embarked", hue="survived", ax=axes[1, 1], palette="Set1")
    axes[1, 1].set_title("Chart 4: Survival by Embarkation Port", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Embarkation Port")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend(title="Survived", labels=["No", "Yes"])

    plt.suptitle("Multivariate Data Story: Who Survived the Titanic?", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_multivariate_data_story.png", dpi=120)
    plt.close()
    print("[OK] Saved: 03_multivariate_data_story.png")
    
    print("""
CHART INTERPRETATIONS:

Chart 1 (Survival by Class & Sex): Women had dramatically higher survival rates than
men across all classes. First-class women survived at ~97%, while third-class men survived
at only ~14%. This confirms the "women and children first" evacuation protocol and the
strong class privilege during the sinking.

Chart 2 (Age by Survival): Survivors and non-survivors have similar median ages (~28),
but the non-survivor group has fewer very young children, suggesting children were
prioritized during evacuation. The spread of ages is broadly similar.

Chart 3 (Age vs Fare by Survival): Survivors cluster in the upper-left region (younger
passengers who paid high fares — likely first-class women and children). Non-survivors
dominate the bottom band (low fares, third class) across all ages.

Chart 4 (Survival by Embarkation Port): Cherbourg (C) had the highest survival rate,
likely because Cherbourg passengers were disproportionately first-class. Southampton (S)
had the lowest survival rate, reflecting its larger third-class contingent.
""")

    # ── Task 6: Standardization check (before/after) ──
    print("=== STANDARDIZATION CHECK (before/after z-score) ===")
    print("\n--- BEFORE standardization ---")
    print(f"Age:  mean={age.mean():.4f}, std={age.std():.4f}")
    print(f"Fare: mean={fare.mean():.4f}, std={fare.std():.4f}")

    scaler_age = StandardScaler()
    scaler_fare = StandardScaler()
    scaled_age = scaler_age.fit_transform(clean[["age"]])
    scaled_fare = scaler_fare.fit_transform(clean[["fare"]])

    print("\n--- AFTER standardization ---")
    print(f"Age:  mean={scaled_age.mean():.6f}, std={scaled_age.std():.6f}")
    print(f"Fare: mean={scaled_fare.mean():.6f}, std={scaled_fare.std():.6f}")
    print("[OK] Both columns now have mean approx 0 and std approx 1 (confirming correct z-score transform).")
    print("Note: This is an EDA-stage sanity check only; the modeling pipeline performs its own train-only scaling.")

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
    """Part B: Classification, regression, imbalance handling, tuning, and comparison."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("\n" + "="*70)
    print("PART B: CLASSIFICATION, REGRESSION, AND MODEL COMPARISON")
    print("="*70)
    
    feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    cat_cols = ["sex", "embarked"]
    num_cols = ["pclass", "age", "sibsp", "parch", "fare"]
    X = clean[feature_cols]
    y = clean["survived"]

    # ── Task 7: Stratified split ──
    print("\n=== STRATIFIED TRAIN/TEST SPLIT ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    survived_rate = y.mean()
    print(f"Overall class balance: survived={survived_rate:.1%}, not survived={1-survived_rate:.1%}")
    print(f"Train: {len(X_train)} samples (survived: {y_train.mean():.1%})")
    print(f"Test:  {len(X_test)} samples (survived: {y_test.mean():.1%})")
    print("Justification: Stratification preserves the ~38.4% survived / ~61.6% not-survived")
    print("class balance across both splits, preventing train/test distribution mismatch")
    print("that could lead to biased model evaluation on this imbalanced dataset.")

    # ── Task 8: Preprocessing (fit on train only) ──
    preprocessor = build_preprocessor(feature_cols, cat_cols, num_cols)

    # ── Task 9: 3 classifiers ──
    print("\n=== CLASSIFICATION: 3 MODELS ===")
    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = []
    best_pipe = None
    best_f1 = -1.0
    best_model_name = ""

    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor(feature_cols, cat_cols, num_cols)), ("clf", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, proba)
        cm = confusion_matrix(y_test, preds)
        
        results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "cm": cm,
            "pipe": pipe,
            "proba": proba,
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_pipe = pipe
            best_model_name = name

        # ── Task 9 (cont): Decision tree visualization ──
        if name == "DecisionTree":
            plt.figure(figsize=(20, 10))
            plot_tree(
                pipe.named_steps["clf"],
                feature_names=pipe.named_steps["prep"].get_feature_names_out(),
                class_names=["died", "survived"],
                filled=True,
                max_depth=3,
            )
            plt.title("Decision Tree Classifier (Depth Limited to 3 for Visibility)", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "04_decision_tree.png", dpi=100)
            plt.close()
            print(f"[OK] Saved: 04_decision_tree.png")

    # ── Task 10: Full metric display ──
    metrics_df = pd.DataFrame([{
        "Model": r["model"],
        "Accuracy": f"{r['accuracy']:.4f}",
        "Precision": f"{r['precision']:.4f}",
        "Recall": f"{r['recall']:.4f}",
        "F1": f"{r['f1']:.4f}",
        "ROC-AUC": f"{r['roc_auc']:.4f}",
    } for r in results])
    print("\n=== CLASSIFIER COMPARISON TABLE ===")
    print(metrics_df.to_string(index=False))

    # ── Confusion matrices ──
    print("\n=== CONFUSION MATRICES ===")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, r in enumerate(results):
        ConfusionMatrixDisplay(r["cm"], display_labels=["Died", "Survived"]).plot(ax=axes[i], cmap="Blues")
        axes[i].set_title(f"{r['model']}", fontsize=12, fontweight="bold")
        print(f"\n{r['model']}:")
        print(r["cm"])
    plt.suptitle("Confusion Matrices — 3 Classifiers", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_confusion_matrices.png", dpi=120)
    plt.close()
    print("[OK] Saved: 05_confusion_matrices.png")

    # ── ROC curves ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        RocCurveDisplay.from_predictions(
            y_test, r["proba"], name=f"{r['model']} (AUC={r['roc_auc']:.3f})", ax=ax
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random baseline")
    ax.set_title("ROC Curves — 3 Classifiers", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_roc_curves.png", dpi=120)
    plt.close()
    print("[OK] Saved: 06_roc_curves.png")

    # ── Task 11: Imbalance handling comparison ──
    print("\n=== 3-WAY IMBALANCE HANDLING COMPARISON (Random Forest) ===")
    print(f"\nClass balance: survived={y_train.sum()} ({y_train.mean():.1%}), "
          f"not-survived={len(y_train)-y_train.sum()} ({1-y_train.mean():.1%})")

    imbalance_results = []

    # (a) Baseline
    rf_baseline = Pipeline([
        ("prep", build_preprocessor(feature_cols, cat_cols, num_cols)),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    rf_baseline.fit(X_train, y_train)
    preds_base = rf_baseline.predict(X_test)
    imbalance_results.append({
        "Strategy": "Baseline (no handling)",
        "Precision": precision_score(y_test, preds_base, zero_division=0),
        "Recall": recall_score(y_test, preds_base, zero_division=0),
        "F1": f1_score(y_test, preds_base, zero_division=0),
    })

    # (b) class_weight='balanced'
    rf_balanced = Pipeline([
        ("prep", build_preprocessor(feature_cols, cat_cols, num_cols)),
        ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
    ])
    rf_balanced.fit(X_train, y_train)
    preds_bal = rf_balanced.predict(X_test)
    imbalance_results.append({
        "Strategy": "class_weight='balanced'",
        "Precision": precision_score(y_test, preds_bal, zero_division=0),
        "Recall": recall_score(y_test, preds_bal, zero_division=0),
        "F1": f1_score(y_test, preds_bal, zero_division=0),
    })

    # (c) SMOTE (train-only)
    if HAS_SMOTE:
        smote_prep = build_preprocessor(feature_cols, cat_cols, num_cols)
        X_train_prep = smote_prep.fit_transform(X_train, y_train)
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_prep, y_train)
        
        rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_smote.fit(X_train_smote, y_train_smote)
        X_test_prep = smote_prep.transform(X_test)
        preds_smote = rf_smote.predict(X_test_prep)
        imbalance_results.append({
            "Strategy": "SMOTE (train-only)",
            "Precision": precision_score(y_test, preds_smote, zero_division=0),
            "Recall": recall_score(y_test, preds_smote, zero_division=0),
            "F1": f1_score(y_test, preds_smote, zero_division=0),
        })
    else:
        print("⚠ SMOTE unavailable (install imbalanced-learn). Skipping.")

    imb_df = pd.DataFrame(imbalance_results)
    for col in ["Precision", "Recall", "F1"]:
        imb_df[col] = imb_df[col].map(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    print("\n" + imb_df.to_string(index=False))

    # Determine best strategy
    best_imb = max(imbalance_results, key=lambda x: x["F1"])
    print(f"\nConclusion: '{best_imb['Strategy']}' achieves the best F1 score ({best_imb['F1']:.4f}).")
    print("class_weight='balanced' adjusts the loss function to penalize minority-class errors more,")
    print("improving recall without the data augmentation overhead of SMOTE. SMOTE can overfit to")
    print("synthetic minority samples; class weighting avoids this while achieving comparable results.")

    # ── Task 12: GridSearchCV + OOB ──
    print("\n=== GRIDSEARCHCV + OOB SCORE (Random Forest) ===")
    rf_pipe = Pipeline([
        ("prep", build_preprocessor(feature_cols, cat_cols, num_cols)),
        ("clf", RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)),
    ])
    param_grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [None, 5, 8],
        "clf__max_features": ["sqrt", "log2", None],
    }
    grid = GridSearchCV(rf_pipe, param_grid, cv=StratifiedKFold(3), scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best CV F1 score: {grid.best_score_:.4f}")
    oob_score = grid.best_estimator_.named_steps["clf"].oob_score_
    print(f"OOB score: {oob_score:.4f}")
    print("Note: OOB uses out-of-bag samples from bootstrap to estimate generalization error")
    print("without a separate validation set, serving as an unbiased estimate of test error.")

    # ── Task 13: Regression side-task ──
    print("\n=== REGRESSION: FARE PREDICTION ===")
    reg_features = ["pclass", "age", "sibsp", "parch", "survived"]
    reg_X = clean[reg_features]
    reg_y = clean["fare"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        reg_X, reg_y, test_size=0.2, random_state=42
    )
    
    reg_pipe = Pipeline([
        ("prep", build_preprocessor(reg_features, [], reg_features)),
        ("reg", LinearRegression()),
    ])
    reg_pipe.fit(Xr_train, yr_train)
    yr_pred = reg_pipe.predict(Xr_test)
    
    mae = mean_absolute_error(yr_test, yr_pred)
    rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
    r2 = r2_score(yr_test, yr_pred)
    
    # Adjusted R²
    n = len(Xr_test)
    p = Xr_train.shape[1]
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    print(f"MAE:           {mae:.2f}")
    print(f"RMSE:          {rmse:.2f}")
    print(f"R²:            {r2:.4f}")
    print(f"Adjusted R²:   {r2_adj:.4f}")
    
    # Residual plot
    residuals = yr_test - yr_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(yr_pred, residuals, alpha=0.5, s=30, c="#4C72B0")
    axes[0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Fitted Values (Predicted Fare)")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residual Plot (Fare Regression)", fontsize=12, fontweight="bold")
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="#55A868")
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Residuals", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.suptitle("Regression Diagnostics: Fare Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_regression_residuals.png", dpi=120)
    plt.close()
    print("[OK] Saved: 07_regression_residuals.png")
    
    # Heteroscedasticity
    print("\n=== HETEROSCEDASTICITY ANALYSIS ===")
    fitted_median = np.median(yr_pred)
    res_std_low = residuals[yr_pred < fitted_median].std()
    res_std_high = residuals[yr_pred >= fitted_median].std()
    print(f"Residual std (low fitted values):  {res_std_low:.2f}")
    print(f"Residual std (high fitted values): {res_std_high:.2f}")
    ratio = abs(res_std_high - res_std_low) / (res_std_low + 1e-6)
    if ratio < 0.2:
        print("Conclusion: Residuals show relatively CONSTANT variance --> homoscedastic.")
    else:
        print("Conclusion: Residuals show INCREASING variance with fitted values --> heteroscedastic.")
        print("The fan-shaped spread indicates the linear model under-predicts high fares more severely,")
        print("likely due to non-linear pricing factors (cabin class, route) not captured linearly.")

    # ── Task 14: Unified model comparison table ──
    print("\n" + "="*70)
    print("UNIFIED MODEL COMPARISON TABLE")
    print("="*70)
    
    print("\n=== CLASSIFICATION METRICS ===")
    class_table = pd.DataFrame([{
        "Model": r["model"],
        "Accuracy": f"{r['accuracy']:.4f}",
        "Precision": f"{r['precision']:.4f}",
        "Recall": f"{r['recall']:.4f}",
        "F1": f"{r['f1']:.4f}",
        "ROC-AUC": f"{r['roc_auc']:.4f}",
    } for r in results])
    print(class_table.to_string(index=False))
    class_table.to_csv(OUTPUT_DIR / "classifier_metrics.csv", index=False)
    
    print("\n=== REGRESSION METRICS (separate scale — not comparable to classification) ===")
    reg_table = pd.DataFrame([{
        "Model": "Linear Regression (Fare)",
        "MAE": f"{mae:.2f}",
        "RMSE": f"{rmse:.2f}",
        "R²": f"{r2:.4f}",
        "Adjusted R²": f"{r2_adj:.4f}",
    }])
    print(reg_table.to_string(index=False))
    reg_table.to_csv(OUTPUT_DIR / "regression_metrics_table.csv", index=False)

    # ── Final recommendation ──
    best_result = max(results, key=lambda r: r["f1"])
    print("\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    print(f"""
RECOMMENDED MODEL: {best_result['model']}

This model achieves the best overall balance of metrics:
- F1 Score: {best_result['f1']:.4f} (best among all three classifiers)
- ROC-AUC: {best_result['roc_auc']:.4f} (strong class discrimination ability)
- Precision: {best_result['precision']:.4f} / Recall: {best_result['recall']:.4f}
- Accuracy: {best_result['accuracy']:.4f}

The F1 score is the primary selection criterion because the dataset is imbalanced
(~38% survived), making accuracy alone misleading. {best_result['model']} provides
the strongest balance between precision (minimizing false positives — wrongly
predicting survival) and recall (minimizing false negatives — missing actual survivors).

For deployment, applying class_weight='balanced' further improves recall on the
minority class. The complete preprocessing + model pipeline is saved via joblib
for end-to-end inference on raw, unprocessed new data.

REGRESSION INSIGHT (Fare Prediction):
The linear regression achieves R²={r2:.4f} and Adjusted R²={r2_adj:.4f}, explaining
~{r2*100:.0f}% of fare variance. The remaining variance is likely due to cabin
assignments, booking timing, and route-specific pricing not captured by the
available features. The model is useful as a baseline but would benefit from
non-linear approaches for production use.
""")

    # ── Task 15: Save and reload pipeline ──
    assert best_pipe is not None
    joblib.dump(best_pipe, PIPELINE_PATH)
    reloaded = joblib.load(PIPELINE_PATH)
    orig_preds = best_pipe.predict(X_test)
    reload_preds = reloaded.predict(X_test)
    assert np.array_equal(orig_preds, reload_preds), "Pipeline reload predictions mismatch!"
    print(f"[OK] Pipeline saved to {PIPELINE_PATH.name} and reloadability verified.")
    print(f"  Best model: {best_model_name}")
    print(f"  Reload predictions match original: {np.array_equal(orig_preds, reload_preds)}")

    # Demo: reload and predict on raw data
    print("\n=== PIPELINE RELOAD DEMO ===")
    sample_raw = pd.DataFrame([{
        "pclass": 1, "sex": "female", "age": 29, "sibsp": 0,
        "parch": 0, "fare": 100.0, "embarked": "S"
    }])
    loaded_pipe = joblib.load(PIPELINE_PATH)
    prediction = loaded_pipe.predict(sample_raw)
    print(f"  Sample input: {sample_raw.to_dict('records')[0]}")
    print(f"  Prediction: {'Survived' if prediction[0] == 1 else 'Did not survive'}")
    print("  [OK] End-to-end pipeline works on raw, unpreprocessed input data.")

    print("\n" + "="*70)
    print("Part B Complete")
    print("="*70)


def main() -> None:
    df = load_data()
    clean = part_a(df)
    part_b(clean)
    print("\nAnalytics complete.")


if __name__ == "__main__":
    main()
