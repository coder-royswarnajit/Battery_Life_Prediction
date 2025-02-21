import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_excel('C:/Users/SWARNAJIT ROY/Desktop/Projects/BatteryPrediction/Battery_Life_Prediction/device_battery_data.xlsx')

# Define helper functions
def get_alpha(category):
    alpha_values = {'Laptop': 0.80, 'Smartwatch': 0.70, 'Tablet': 0.70, 'Smartphone': 0.75, 'Gaming console': 0.85}
    return alpha_values.get(category, 0.75)

def get_beta(category):
    beta_values = {'Laptop': 0.20, 'Smartwatch': 0.30, 'Tablet': 0.30, 'Smartphone': 0.25, 'Gaming console': 0.15}
    return beta_values.get(category, 0.25)

def get_temperature_factor(temp):
    try:
        temp = float(temp)
    except (ValueError, TypeError):
        return 1.0
    if 20 <= temp <= 30:
        return 1.0
    elif temp > 35:
        return 0.85
    elif temp < 0:
        return 0.65
    else:
        return 1.0

# Feature Engineering
data['alpha'] = data['Category'].apply(get_alpha)
data['beta'] = data['Category'].apply(get_beta)

data['Battery_Degradation_Factor'] = (
    1 - (data['alpha'] * (data['Charging_Cycles'] / data['Maximum Cycles'])) -
    (data['beta'] * (data['Device Age (years)'] / data['Battery Lifespan (years)']))
)

data['Average_Power_Consumption'] = (
    (data['Active Time per Day (hours)'] * data['Active Power Consumption (mW)']) +
    (data['Sleep Time per Day (hours)'] * data['Sleep Power Consumption (mW)'])
) / 24

data['Battery_Capacity_mWh'] = data['Battery Capacity (mAh)'] * data['Battery Voltage (V)']

data['Operating Temp (°C)'] = pd.to_numeric(data['Operating Temp (°C)'], errors='coerce')
data['Temperature Factor'] = data['Operating Temp (°C)'].apply(get_temperature_factor)

env_map = {'ideal': 1.0, 'low humidity/dust': 0.95, 'high humidity/dust': 0.925, 'extreme': 0.85}
data['Environmental Conditions'] = data['Environmental Conditions'].str.lower().str.strip()
data['Environmental Factor'] = data['Environmental Conditions'].map(env_map).fillna(1.0)

data['Battery_Life'] = (
    data['Battery_Capacity_mWh'] * data['Battery_Degradation_Factor'] *
    data['Temperature Factor'] * data['Environmental Factor']
) / data['Average_Power_Consumption']

# Save processed data
output_file = 'C:/Users/SWARNAJIT ROY/Desktop/Projects/BatteryPrediction/Battery_Life_Prediction/Updated_Battery_Life.xlsx'
data.to_excel(output_file, index=False)
print(f"Battery Life has been saved to {output_file}")

# Model Training
features = ['Battery_Degradation_Factor', 'Average_Power_Consumption', 'Battery_Capacity_mWh', 'Temperature Factor', 'Environmental Factor']
target = 'Battery_Life'

X = data[features]
y = data[target]

if len(X) < 2:
    raise ValueError("Not enough samples in the dataset to perform a train/test split.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Calculate performance metrics
def evaluate_model(name, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} Model:")
    print(f"  MSE: {mse:.4f}, R²: {r2:.4f}\n")
    return mse, r2

evaluate_model("Gradient Boosting", y_test, gb_pred)
evaluate_model("XGBoost", y_test, xgb_pred)

# Hybrid Ensemble Model (Averaging predictions)
ensemble_pred = (gb_pred + xgb_pred) / 2
evaluate_model("Ensembled Hybrid Model", y_test, ensemble_pred)

# Display actual vs predicted values
print("Actual vs Predicted (First 10 values):")
comparison_df = pd.DataFrame({
    'Actual': y_test[:10].values,
    'GB Predicted': gb_pred[:10],
    'XGB Predicted': xgb_pred[:10],
    'Hybrid Model': ensemble_pred[:10]
})
print(comparison_df)
