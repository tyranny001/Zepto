import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('analytics/titanic.csv')
clean = df.drop_duplicates().copy()
clean["age"] = clean["age"].fillna(clean["age"].median())
clean["survived"] = clean["survived"].astype(int)

feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = clean[feature_cols]
y = clean["survived"]

# Check stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("CRITERION 10: Stratified train/test split")
print(f"Total samples: {len(y)}")
print(f"Train size: {len(y_train)} ({len(y_train)/len(y)*100:.1f}%)")
print(f"Test size: {len(y_test)} ({len(y_test)/len(y)*100:.1f}%)")
print(f"\nSurvival rate in full dataset: {y.mean():.4f}")
print(f"Survival rate in train: {y_train.mean():.4f}")
print(f"Survival rate in test: {y_test.mean():.4f}")
print(f"Stratify parameter used: True")
