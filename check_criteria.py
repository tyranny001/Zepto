import pandas as pd
import numpy as np

df = pd.read_csv('analytics/titanic.csv')

# Criterion 2: Missing value percentages
print('CRITERION 2: Missing values')
missing = df.isnull().sum()
for col in missing[missing > 0].index:
    pct = (missing[col] / len(df)) * 100
    print(f'  {col}: {missing[col]} ({pct:.1f}%)')

# Criterion 3: IQR outlier analysis
print('\nCRITERION 3: IQR Outlier Analysis')
for col in ['age', 'fare']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)][col].dropna()
    print(f'  {col}: {len(outliers)} outliers')
    print(f'    Skewness: {df[col].skew():.2f}')

# Criterion 4: Bivariate survival rates
print('\nCRITERION 4: Bivariate survival rates')
print('  By sex:')
print(df.groupby('sex')['survived'].mean())
print('  By pclass:')
print(df.groupby('pclass')['survived'].mean())

# Criterion 5&6&7: Correlation analysis
print('\nCRITERION 5,6,7: Correlation Analysis')
numeric = df.select_dtypes(include=[np.number])
corr = numeric.corr()
print(f'  Shape: {corr.shape}')
print(f'  Columns: {list(corr.columns)}')

# Get top 2 correlations
corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        col1 = corr.columns[i]
        col2 = corr.columns[j]
        val = corr.iloc[i, j]
        corr_pairs.append((f'{col1}-{col2}', val))

corr_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
print(f'  Top 2 correlations:')
for name, val in corr_pairs[:2]:
    print(f'    {name}: {val:.4f}')
