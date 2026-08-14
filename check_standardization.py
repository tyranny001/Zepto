import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('analytics/titanic.csv')

# Clean data
clean = df.drop_duplicates().copy()
clean["age"] = clean["age"].fillna(clean["age"].median())
clean["embarked"] = clean["embarked"].fillna(clean["embarked"].mode()[0])
clean["embark_town"] = clean["embark_town"].fillna(clean["embark_town"].mode()[0])
clean["deck"] = clean["deck"].fillna("Unknown")

print('CRITERION 9: Standardization check')
print('\nAge:')
scaler_age = StandardScaler()
age_scaled = scaler_age.fit_transform(clean[["age"]])
print(f'  Mean: {age_scaled.mean():.6f}')
print(f'  Std: {age_scaled.std():.6f}')

print('\nFare:')
scaler_fare = StandardScaler()
fare_scaled = scaler_fare.fit_transform(clean[["fare"]])
print(f'  Mean: {fare_scaled.mean():.6f}')
print(f'  Std: {fare_scaled.std():.6f}')
