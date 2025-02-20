import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_excel('C:/Users/SWARNAJIT ROY/Desktop/Projects/BatteryPrediction/device_battery_data.xlsx')

def get_alpha(category):
    alpha_values = {
        'Laptop': 0.80, 'Smartwatch': 0.70, 'Tablet': 0.70, 
        'Smartphone': 0.75, 'Gaming console': 0.85
    }
    return alpha_values.get(category, 0.75) 

def get_beta(category):
    beta_values = {
        'Laptop': 0.20, 'Smartwatch': 0.30, 'Tablet': 0.30, 
        'Smartphone': 0.25, 'Gaming console': 0.15
    }
    return beta_values.get(category, 0.25)  

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

data['Operating Temp (°C)'] = pd.to_numeric(data['Operating Temp (°C)'], errors='coerce')
data['Temperature Factor'] = data['Operating Temp (°C)'].apply(get_temperature_factor)


env_map = {
    'ideal': 1.0,
    'low humidity/dust': 0.95,
    'high humidity/dust': 0.925,
    'extreme': 0.85
}
data['Environmental Conditions'] = data['Environmental Conditions'].str.lower().str.strip()
data['Environmental Factor'] = data['Environmental Conditions'].map(env_map).fillna(1.0)


data['Battery_Life'] = (
    data['Battery_Capacity_mWh'] * data['Battery_Degradation_Factor'] *
    data['Temperature Factor'] * data['Environmental Factor']
) / data['Average_Power_Consumption']


output_file = 'C:/Users/SWARNAJIT ROY/Desktop/Projects/BatteryPrediction/Updated_Battery_Life.xlsx'
data.to_excel(output_file, index=False)
print(f"Battery Life has been saved to {output_file}")


features = [
    'Battery_Degradation_Factor', 'Average_Power_Consumption',
    'Battery_Capacity_mWh', 'Temperature Factor', 'Environmental Factor'
]
target = 'Battery_Life'

X = data[features]
y = data[target]

# Ensure enough samples for train/test split
if len(X) < 2:
    raise ValueError("Not enough samples in the dataset to perform a train/test split.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print("Linear Regression Model:")
print(f"  MSE: {lr_mse:.4f}, R²: {lr_r2:.4f}\n")

# Gradient Boosting Model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

gb_mse = mean_squared_error(y_test, gb_pred)
gb_r2 = r2_score(y_test, gb_pred)
print("Gradient Boosting Model:")
print(f"  MSE: {gb_mse:.4f}, R²: {gb_r2:.4f}\n")

# Hybrid Ensemble Model (Averaging predictions)
ensemble_pred = (lr_pred + gb_pred) / 2
ensemble_mse = mean_squared_error(y_test, ensemble_pred)
ensemble_r2 = r2_score(y_test, ensemble_pred)
print("Ensembled Hybrid Model (Average of LR & GB):")
print(f"  MSE: {ensemble_mse:.4f}, R²: {ensemble_r2:.4f}\n")

# Compute SSE and SST
sse_lr = np.sum((y_test - lr_pred) ** 2)
sst_lr = np.sum((y_test - np.mean(y_test)) ** 2)
print(f"Linear Regression SSE: {sse_lr:.4f}, SST: {sst_lr:.4f}, R²: {1 - (sse_lr / sst_lr):.4f}")

sse_gb = np.sum((y_test - gb_pred) ** 2)
sst_gb = np.sum((y_test - np.mean(y_test)) ** 2)
print(f"Gradient Boosting SSE: {sse_gb:.4f}, SST: {sst_gb:.4f}, R²: {1 - (sse_gb / sst_gb):.4f}")

# Display actual vs predicted values
print("Actual vs Predicted (First 10 values):")
comparison_df = pd.DataFrame({
    'Actual': y_test[:10].values,
    'LR Predicted': lr_pred[:10],
    'GB Predicted': gb_pred[:10]
})
print(comparison_df)
