import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# --- 1. Prepare the Data ---
# Define the dataset using  data

df = pd.read_csv("BuckData.csv") 

# Convert monetary values: Remove "$" and convert to float
money_columns = ["Average CPC", "Spend", "CPA"]
df[money_columns] = df[money_columns].replace('[\$,]', '', regex=True).astype(float)

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Convert percentages: Remove "%" and convert to float
percent_columns = ["CTR", "Conversion Rate"]
df[percent_columns] = df[percent_columns].replace('%', '', regex=True).astype(float) / 100
# Convert numerical columns with commas to float
columns_to_clean = ["Impressions", "Spend", "CPA"]
for col in columns_to_clean:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)

#Check column names before dropping**
print("Available Columns:", df.columns)  # Debugging step

# Drop unnecessary columns
drop_cols = ["Impressions"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")


# Convert the Month column to datetime
df['Month_dt'] = pd.to_datetime(df['Month'], format='%b-%y')
# Extract the month number (1-12)
df['Month_num'] = df['Month_dt'].dt.month
# Create cyclical features for month using sine and cosine transformations
df['month_sin'] = np.sin(2 * np.pi * df['Month_num'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['Month_num'] / 12)

# --- 2. Define the Model Inputs ---
# cyclical month features.
# The coefficient for Spend will indicate its weighted influence on leads.
X = df[['Spend', 'month_sin', 'month_cos']]
y = df['Leads']

# --- 3. Build and Train the Linear Regression Model ---
# Creates pipeline that standardizes  features and fits a Linear Regression model.
pipeline = make_pipeline(StandardScaler(), LinearRegression())
pipeline.fit(X, y)

# Retrieve trained linear regression model
model = pipeline.named_steps['linearregression']
print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef}")




# --- 4. Make Predictions and Visualize ---
df['Predicted_Leads'] = pipeline.predict(X)

plt.figure(figsize=(10, 6))
plt.plot(df['Month_dt'], df['Leads'], marker='o', label='Actual Leads')
plt.plot(df['Month_dt'], df['Predicted_Leads'], marker='x', label='Predicted Leads')
plt.xlabel('Month')
plt.ylabel('Leads')
plt.title('Actual vs Predicted Leads')
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error

# Compute MAE
mae = mean_absolute_error(df['Leads'], df['Predicted_Leads'])
print("Mean Absolute Error:", mae)