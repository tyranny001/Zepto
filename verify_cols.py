import pandas as pd
import numpy as np

df = pd.read_csv('analytics/titanic.csv')

# Get numeric columns
numeric = df.select_dtypes(include=[np.number])

print('Numeric columns in df:', list(numeric.columns))
print('Expected columns: [survived, pclass, age, sibsp, parch, fare]')
print('Match:', set(numeric.columns) == {'"survived"', '"pclass"', '"age"', '"sibsp"', '"parch"', '"fare"'})

# Check adult_male and alone
print('\nColumns with bool dtype:')
bool_cols = df.select_dtypes(include=[bool]).columns.tolist()
print(bool_cols)

print('\nTotal columns:', df.shape[1])
print('Column dtypes:')
for col in df.columns:
    print(f'  {col}: {df[col].dtype}')
