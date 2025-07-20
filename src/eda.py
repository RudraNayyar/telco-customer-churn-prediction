import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)


DATA_PATH = os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')


df = pd.read_csv(DATA_PATH)


print('Sample data:')
print(df.head())


print(f'\nDataset contains {df.shape[0]} rows and {df.shape[1]} columns.')


print('\nMissing values per column:')
print(df.isnull().sum())


print('\nData info:')
df.info()


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn', palette='Set2')
plt.title('Distribution of Churned vs. Non-Churned Customers')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()

