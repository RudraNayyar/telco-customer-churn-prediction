import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

# Define the path to the dataset
DATA_PATH = os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Show the first few rows to get a sense of the data
print('Sample data:')
print(df.head())

# Display the shape of the dataset
print(f'\nDataset contains {df.shape[0]} rows and {df.shape[1]} columns.')

# Check for missing values
print('\nMissing values per column:')
print(df.isnull().sum())

# Show data types and non-null counts
print('\nData info:')
df.info()

# Plot the distribution of the target variable (Churn)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn', palette='Set2')
plt.title('Distribution of Churned vs. Non-Churned Customers')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()

# Optional: Save the plot
# plt.savefig('../models/churn_distribution.png') 