import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
raw_data = pd.read_csv(r"ai4i2020 (1).csv")

# Prepare the data
X = raw_data.drop(columns=['UDI', 'Product ID', 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', ])
y = raw_data['Machine failure'].values


correlations = X.corr()['Machine failure'].drop(["Machine failure"]) # Exclude 'module_time' itself

print(correlations)

# Plot the correlations
plt.figure(figsize=(8, 6))
sns.barplot(x=correlations.index, y=correlations.values, palette='coolwarm')

# Add labels and title
plt.xlabel('Features')
plt.ylabel('Correlation with machine failure')
plt.title('Correlation between machine failure and other feature')
plt.xticks(rotation=45)
plt.show()