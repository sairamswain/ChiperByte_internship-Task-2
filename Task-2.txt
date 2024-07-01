#importing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the datasets
file_path1 = "C:/User/sDELL/OneDrive/Desktop/Unemployment in India (1).xlsx"
file_path2 = "C:/Users/DELL/OneDrive/Desktop/Unemployment_Rate_upto_11_2020 (1).xlsx"

df1 = pd.read_excel("Unemployment in India (1).xlsx")
df2 = pd.read_excel("Unemployment in India (1).xlsx")

# Display the first few rows of each DataFrame to verify successful loading
print("Dataset 1:\n", df1.head())
print("\nDataset 2:\n", df2.head())

# Concatenate the DataFrames row-wise (you can choose column-wise if needed)
df_merged = pd.concat([df1, df2], ignore_index=True)

# Convert the 'Date' column to datetime format
df_merged['Date'] = pd.to_datetime(df_merged['Date'])

# Display dataset summary
print("\nDataset Summary:")
print(df_merged.describe())

# Group by Region and calculate mean unemployment rate for each region
region_mean = df_merged.groupby('Region')['Estimated Unemployment Rate (%)'].mean().reset_index()

# Create a bar graph
plt.figure(figsize=(12, 6))
plt.bar(region_mean['Region'], region_mean['Estimated Unemployment Rate (%)'], color='skyblue')
plt.xlabel('Region')
plt.ylabel('Mean Unemployment Rate (%)')
plt.title('Mean Unemployment Rate by Region')
plt.xticks(rotation=45)
plt.show()

# Create a histogram
plt.figure(figsize=(12, 6))
plt.hist(df_merged['Estimated Unemployment Rate (%)'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Estimated Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Estimated Unemployment Rates')
plt.show()

# Group by Region and calculate mean unemployment rate for each region
region_mean = df_merged.groupby('Region')['Estimated Unemployment Rate (%)'].mean().reset_index()

# Sort by unemployment rate descending and select top 10
top_10_regions = region_mean.nlargest(10, 'Estimated Unemployment Rate (%)')

#Top 10 Regions
plt.figure(figsize=(10, 8))
plt.pie(top_10_regions['Estimated Unemployment Rate (%)'], labels=top_10_regions['Region'], autopct='%1.1f%%', startangle=140)
plt.title('Top 10 Regions with Highest Unemployment Rate')
plt.axis('equal')  
plt.show()

plt.figure(figsize=(12, 6))
plt.hist(df_merged['Estimated Unemployment Rate (%)'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Estimated Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Estimated Unemployment Rates')
plt.show()

df_merged.columns


# Selecting features (X)
X = df_merged[['Region', 'Date', 'Frequency', 'Estimated Employed', 'Estimated Labour Participation Rate (%)', 'Area']]

# Selecting the target variable (y)
y = df_merged['Estimated Unemployment Rate (%)']

# Split the data into training and testing sets (adjust test_size and random_state as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

df_merged.head()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# Assuming 'df' contains your dataset snippet
# Selecting features (X) and target variables (y) for both Linear and Logistic Regression
X_linear = df_merged[['Estimated Employed', 'Estimated Labour Participation Rate (%)']]
y_linear = df_merged['Estimated Unemployment Rate (%)']

# Example of handling missing values by imputing with mean
X_linear.fillna(X_linear.mean(), inplace=True)
y_linear.fillna(y_linear.mean(), inplace=True)

# For Logistic Regression example (creating a binary target variable for classification)
threshold = 5
df_merged['Above_Threshold'] = np.where(df_merged['Estimated Unemployment Rate (%)'] > threshold, 1, 0)
X_logistic = df_merged[['Estimated Employed', 'Estimated Labour Participation Rate (%)']]
y_logistic = df_merged['Above_Threshold']

# Example of handling missing values for logistic regression
X_logistic.fillna(X_logistic.mean(), inplace=True)

# Split the data into training and testing sets
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(X_logistic, y_logistic, test_size=0.2, random_state=42)

# Initialize and train the models
# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_linear, y_train_linear)
y_pred_linear = linear_model.predict(X_test_linear)

# Evaluate Linear Regression model
mse_linear = mean_squared_error(y_test_linear, y_pred_linear)
r2_linear = r2_score(y_test_linear, y_pred_linear)
print(f"Linear Regression Mean Squared Error: {mse_linear:.2f}")
print(f"Linear Regression R-squared: {r2_linear:.2f}")

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train_logistic, y_train_logistic)
y_pred_logistic = logistic_model.predict(X_test_logistic)

# Evaluate Logistic Regression model
accuracy_logistic = accuracy_score(y_test_logistic, y_pred_logistic)
print(f"Logistic Regression Accuracy: {accuracy_logistic:.2f}")
print("Classification Report:")
print(classification_report(y_test_logistic, y_pred_logistic))
