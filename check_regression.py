import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('analytics/titanic.csv')

# Clean data
clean = df.drop_duplicates().copy()
clean["age"] = clean["age"].fillna(clean["age"].median())
clean["embarked"] = clean["embarked"].fillna(clean["embarked"].mode()[0])
clean["embark_town"] = clean["embark_town"].fillna(clean["embark_town"].mode()[0])
clean["deck"] = clean["deck"].fillna("Unknown")
clean["survived"] = clean["survived"].astype(int)

# Criterion 17: Regression task - predict fare
print('CRITERION 17: Regression task (predict fare)')
reg_X = clean[["pclass", "age", "sibsp", "parch", "survived"]]
reg_y = clean["fare"]

reg_pipe = Pipeline([
    ("prep", ColumnTransformer(
        transformers=[
            ("num",
             Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
             ["pclass", "age", "sibsp", "parch", "survived"]),
        ],
        remainder="drop",
    )),
    ("reg", LinearRegression()),
])

Xr_train, Xr_test, yr_train, yr_test = train_test_split(reg_X, reg_y, test_size=0.2, random_state=42)
reg_pipe.fit(Xr_train, yr_train)

yr_pred = reg_pipe.predict(Xr_test)
mse = mean_squared_error(yr_test, yr_pred)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(yr_test - yr_pred))
r2 = r2_score(yr_test, yr_pred)

# Adjusted R2
n = len(yr_test)
p = 5  # number of features
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f'  MAE: {mae:.2f}')
print(f'  RMSE: {rmse:.2f}')
print(f'  R2: {r2:.4f}')
print(f'  Adjusted R2: {adj_r2:.4f}')

# Residuals
residuals = yr_test - yr_pred
print(f'  Residuals mean: {residuals.mean():.4f}')
print(f'  Residuals std: {residuals.std():.2f}')
print(f'  Residuals heteroscedasticity check (residuals vs predictions):')
print(f'    Max residual: {residuals.abs().max():.2f}')
print(f'    Min residual: {residuals.min():.2f}')
