import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils.validation import check_is_fitted

# --- 1. Load and Prepare Data ---
df = pd.read_csv("BuckData.csv")

# ✅ Convert monetary values: Remove "$" and convert to float
money_columns = ["Average CPC", "Spend", "CPA"]
df[money_columns] = df[money_columns].replace('[\$,]', '', regex=True).astype(float)

# ✅ Convert percentages: Remove "%" and convert to float
percent_columns = ["CTR", "Conversion Rate"]
df[percent_columns] = df[percent_columns].replace('%', '', regex=True).astype(float) / 100

# ✅ Convert Month column to datetime and then to numeric format (YYYY.MM)
df['Month_dt'] = pd.to_datetime(df['Month'], format='%b-%y')
df['Month'] = df['Month_dt'].dt.strftime('%Y.%m').astype(float)

# ✅ Convert numerical columns with commas to float
columns_to_clean = ["Impressions", "Spend", "CPA"]
for col in columns_to_clean:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)


# ✅ Drop unnecessary columns
drop_cols = ["Impressions"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

# --- 2. Create Additional Features ---
df["Cost_Per_Lead"] = df["Spend"] / (df["Leads"] + 1)  # Avoid division by zero
df["CPA_Ratio"] = df["CPA"] / df["Spend"]
df["Spend_CTR"] = df["Spend"] * df["CTR"]
df["Log_Spend"] = np.log1p(df["Spend"])  # log1p avoids log(0) issues
df["Log_CPA"] = np.log1p(df["CPA"])

# ✅ Normalize sample weights
df["Weight"] = df["Spend"] / df["Spend"].mean()

# --- 3. Define Features (X) and Target (y) ---
X = df.drop(columns=["Leads", "Month_dt"])
y = df["Leads"]

# ✅ Handle NaN values
y = y.fillna(0)

# ✅ Print dataset details
print(f"Filtered dataset size: {df.shape[0]} rows")
print("Features used in the model:", X.columns)

# --- 4. Manual Train-Test Split (80% Training, 20% Testing) ---
split_index = int(len(df) * 0.8)  # Get 80% index position

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# --- 5. Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 6. Feature Selection ---
selector = RFE(LinearRegression(), n_features_to_select=5)
selector.fit(X_train_scaled, y_train)
print("Selected Features:", X.columns[selector.support_])

# --- 7. Train the Model with Hyperparameter Tuning ---
param_grid = {"fit_intercept": [True, False]}
grid = GridSearchCV(LinearRegression(), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

# ✅ Select the best model
best_model = grid.best_estimator_
best_model.fit(X_train_scaled, y_train, sample_weight=df.loc[X_train.index, "Weight"])

# ✅ Ensure the model is fitted
try:
    check_is_fitted(best_model)
except:
    print("Model is not fitted! Training now...")
    best_model.fit(X_train_scaled, y_train)

# --- 8. Make Predictions ---
y_pred_best = best_model.predict(X_test_scaled)
df["Predicted_Leads"] = best_model.predict(scaler.transform(X))

# --- 9. Evaluate Performance ---
mae = mean_absolute_error(y_test, y_pred_best)
r2 = r2_score(y_test, y_pred_best)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared Score: {r2:.2f}")


# --- 10. Plot Actual vs. Predicted Leads ---

# Ensure data is sorted by Month_dt
df = df.sort_values(by="Month_dt")

plt.figure(figsize=(10, 6))

# Plot Actual Leads over Time (Month_dt)
plt.plot(df['Month_dt'], df['Leads'], marker='o', color='blue', label='Actual Leads')

# Plot Predicted Leads over Time (Month_dt)
plt.plot(df['Month_dt'], df['Predicted_Leads'], marker='x', linestyle='--', color='orange', label='Predicted Leads')

# Fit and plot the Best-Fit Line (Regression Line) on the actual leads
m, b = np.polyfit(pd.to_numeric(df['Month_dt']), df['Leads'], 1)  # Linear regression
plt.plot(df['Month_dt'], m*pd.to_numeric(df['Month_dt']) + b, color='red', linestyle='--', label='Best-Fit Line')

# Labels and Title
plt.xlabel("Month")
plt.ylabel("Leads")
plt.title("Actual vs Predicted Leads with Best-Fit Line")
plt.legend()
plt.grid(True)

# Show the plot with rotated x-axis labels for better readability
plt.xticks(rotation=45)
plt.show()


# --- 11. Save Predictions ---
results_df = pd.DataFrame({"Actual Leads": y_test, "Predicted Leads": y_pred_best})
results_df.to_csv("best_fit_line.csv", index=False)
print("Predictions saved as lead_predictions.csv!")

# --- 12. Save the Trained Model ---
import pickle
model_filename = "lead_prediction_model.pkl"
with open(model_filename, "wb") as model_file:
    pickle.dump(best_model, model_file)

print(f"✅ Model saved successfully as {model_filename}")
