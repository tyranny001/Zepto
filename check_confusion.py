import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv('analytics/titanic.csv')

# Clean data
clean = df.drop_duplicates().copy()
clean["age"] = clean["age"].fillna(clean["age"].median())
clean["embarked"] = clean["embarked"].fillna(clean["embarked"].mode()[0])
clean["embark_town"] = clean["embark_town"].fillna(clean["embark_town"].mode()[0])
clean["deck"] = clean["deck"].fillna("Unknown")
clean["survived"] = clean["survived"].astype(int)

feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
cat_cols = ["sex", "embarked"]
num_cols = ["pclass", "age", "sibsp", "parch", "fare"]
X = clean[feature_cols]
y = clean["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
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

models = {
    "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
}

print('CRITERION 14: Confusion matrices for all classifiers')
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    print(f'\n{name}:')
    print(cm)
    print(f'  TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}')
