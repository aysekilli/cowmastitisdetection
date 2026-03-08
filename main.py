import pandas as pd

df = pd.read_csv('cow_milk_mastitis_dataset.csv')
print(df.head())
print(df.shape)
print(df.columns.tolist())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df['class1'].value_counts())