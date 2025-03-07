import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from qiskit_aer import AerSimulator
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient

# Load Data
data = pd.read_excel('C:/Users/SWARNAJIT ROY/Desktop/Projects/BatteryPrediction/Battery_Life_Prediction/device_battery_data.xlsx')

# Feature Engineering
data['alpha'] = data['Category'].map({'Laptop': 0.80, 'Smartwatch': 0.70, 'Tablet': 0.70, 'Smartphone': 0.75, 'Gaming Console': 0.85}).fillna(0.75)
data['beta'] = data['Category'].map({'Laptop': 0.20, 'Smartwatch': 0.30, 'Tablet': 0.30, 'Smartphone': 0.25, 'Gaming Console': 0.15}).fillna(0.25)

data['Battery_Degradation_Factor'] = (1 - (data['alpha'] * (data['Charging_Cycles'] / data['Maximum Cycles'])) - (data['beta'] * (data['Device Age (years)'] / data['Battery Lifespan (years)'])))
data['Average_Power_Consumption'] = ((data['Active Time per Day (hours)'] * data['Active Power Consumption (mW)']) + (data['Sleep Time per Day (hours)'] * data['Sleep Power Consumption (mW)'])) / 24
data['Battery_Capacity_mWh'] = data['Battery Capacity (mAh)'] * data['Battery Voltage (V)']

def get_temperature_factor(temp):
    if pd.isna(temp):
        return 1.0
    temp = float(temp)
    return 1.0 if 20 <= temp <= 30 else 0.85 if temp > 35 else 0.65 if temp < 0 else 1.0

data['Temperature Factor'] = data['Operating Temp (°C)'].apply(get_temperature_factor)
env_map = {'ideal': 1.0, 'low humidity/dust': 0.95, 'high humidity/dust': 0.925, 'extreme': 0.85}
data['Environmental Factor'] = data['Environmental Conditions'].str.lower().str.strip().map(env_map).fillna(1.0)

# Ensure no division by zero
data.loc[:, 'Average_Power_Consumption'] = data['Average_Power_Consumption'].replace(0, np.nan)
data.dropna(subset=['Average_Power_Consumption'], inplace=True)

# Compute Battery_Life if missing
data['Battery_Life'] = (
    data['Battery_Capacity_mWh'] * data['Battery_Degradation_Factor'] *
    data['Temperature Factor'] * data['Environmental Factor'] /
    data['Average_Power_Consumption']
)

data.dropna(subset=['Battery_Life'], inplace=True)

# Remove invalid values before applying MinMax scaling
data = data[data['Battery_Life'] > 0]
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(data[['Battery_Life']].values)

# Prepare Data
features = ['Battery_Degradation_Factor', 'Average_Power_Consumption', 'Battery_Capacity_mWh', 'Temperature Factor', 'Environmental Factor']
X = data[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use PCA dynamically to retain at least 5 features
pca = PCA(n_components=min(5, X.shape[1]))
X_reduced = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42, shuffle=True)

# Quantum Neural Network
seed = 42
algorithm_globals.random_seed = seed
estimator = StatevectorEstimator()
num_qubits = max(1, X_reduced.shape[1])  # Ensure at least 1 qubit

feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement="full")
ansatz = EfficientSU2(num_qubits, reps=2, entanglement="full")

gradient = ParamShiftEstimatorGradient(estimator)

print("Feature Map Parameters:", list(feature_map.parameters))
print("Ansatz Parameters:", list(ansatz.parameters))
print("Combined Circuit:")
print(feature_map.compose(ansatz).draw())

qnn = EstimatorQNN(
    circuit=feature_map.compose(ansatz),
    input_params=feature_map.parameters,  # Use feature_map parameters directly
    weight_params=ansatz.parameters,  # Use ansatz parameters directly
    estimator=estimator,
    gradient=gradient
)

optimizer = SPSA(maxiter=2000, learning_rate=0.02, perturbation=0.01)
nn_regressor = NeuralNetworkRegressor(qnn, optimizer=optimizer)

# Debugging prints
print("X_train shape:", X_train.shape)
print("Number of Qubits:", num_qubits)
print("Feature Map Dimension:", feature_map.num_qubits)

# Cross-validation
kf = KFold(n_splits=2, shuffle=True, random_state=seed)
batch_size = 100  # Increase batch size
r2_scores, mse_scores = [], []

for train_idx, val_idx in kf.split(X_train):
    X_cv_train, X_cv_val = X_train[train_idx][:batch_size], X_train[val_idx][:batch_size]
    y_cv_train, y_cv_val = y_train[train_idx][:batch_size], y_train[val_idx][:batch_size]
    
    nn_regressor.fit(X_cv_train, y_cv_train)
    y_cv_pred = np.maximum(0, y_scaler.inverse_transform(nn_regressor.predict(X_cv_val)))  # Reverse scaling & prevent negatives
    
    mse_scores.append(mean_squared_error(y_cv_val, y_cv_pred))
    r2_scores.append(r2_score(y_cv_val, y_cv_pred))

# Final Model Evaluation
y_pred = np.maximum(0, y_scaler.inverse_transform(nn_regressor.predict(X_test)))  # Reverse scaling & prevent negatives
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Quantum Neural Network Regressor:")
print("  MSE:", mse, "(Avg CV MSE:", np.mean(mse_scores), ")")
print("  R²:", r2, "(Avg CV R²:", np.mean(r2_scores), ")")

print("Actual:", y_scaler.inverse_transform(y_test[:10].reshape(-1, 1)))
print("Predicted:", y_pred[:10])
