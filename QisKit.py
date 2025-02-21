import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from qiskit_aer import Aer
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.gradients import FiniteDiffEstimatorGradient

# Load Data
data = pd.read_excel('C:/Users/SWARNAJIT ROY/Desktop/Projects/BatteryPrediction/Battery_Life_Prediction/device_battery_data.xlsx')

# Feature Engineering
data['alpha'] = data['Category'].map({
    'Laptop': 0.80, 'Smartwatch': 0.70, 'Tablet': 0.70, 
    'Smartphone': 0.75, 'Gaming Console': 0.85
}).fillna(0.75)

data['beta'] = data['Category'].map({
    'Laptop': 0.20, 'Smartwatch': 0.30, 'Tablet': 0.30, 
    'Smartphone': 0.25, 'Gaming Console': 0.15
}).fillna(0.25)

data['Battery_Degradation_Factor'] = (
    1 - (data['alpha'] * (data['Charging_Cycles'] / data['Maximum Cycles'])) -
    (data['beta'] * (data['Device Age (years)'] / data['Battery Lifespan (years)']))
)

data['Average_Power_Consumption'] = ((data['Active Time per Day (hours)'] * data['Active Power Consumption (mW)']) + 
                                      (data['Sleep Time per Day (hours)'] * data['Sleep Power Consumption (mW)'])) / 24

data['Battery_Capacity_mWh'] = data['Battery Capacity (mAh)'] * data['Battery Voltage (V)']

def get_temperature_factor(temp):
    if pd.isna(temp):
        return 1.0
    temp = float(temp)
    if 20 <= temp <= 30:
        return 1.0
    elif temp > 35:
        return 0.85
    elif temp < 0:
        return 0.65
    return 1.0

data['Temperature Factor'] = data['Operating Temp (°C)'].apply(get_temperature_factor)

env_map = {'ideal': 1.0, 'low humidity/dust': 0.95, 'high humidity/dust': 0.925, 'extreme': 0.85}
data['Environmental Factor'] = data['Environmental Conditions'].str.lower().str.strip().map(env_map).fillna(1.0)

data['Battery_Life'] = (
    data['Battery_Capacity_mWh'] * data['Battery_Degradation_Factor'] * 
    data['Temperature Factor'] * data['Environmental Factor'] / data['Average_Power_Consumption']
)

data.dropna(inplace=True)  # Remove missing values

# Prepare Data
features = ['Battery_Degradation_Factor', 'Average_Power_Consumption', 'Battery_Capacity_mWh', 'Temperature Factor', 'Environmental Factor']
target = 'Battery_Life'
X = data[features].values
y = data[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use more PCA components to retain more information
pca = PCA(n_components=min(5, X_scaled.shape[1]))  # Adjust components
X_reduced = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Quantum Neural Network
seed = 42
algorithm_globals.random_seed = seed

estimator = StatevectorEstimator()
gradient = FiniteDiffEstimatorGradient(estimator, epsilon=1e-4)

num_qubits = X_reduced.shape[1]
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=3, entanglement='full')
ansatz = EfficientSU2(num_qubits, reps=3, entanglement='full')

qnn = EstimatorQNN(
    circuit=feature_map.compose(ansatz),
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    estimator=estimator,
    gradient=gradient
)

optimizer = SPSA(maxiter=500)  # Increase iterations for better training
nn_regressor = NeuralNetworkRegressor(qnn, optimizer=optimizer)
nn_regressor.fit(X_train, y_train)
y_pred = nn_regressor.predict(X_test)

# Ensure non-negative predictions
y_pred = np.maximum(0, y_pred)

# Performance Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Quantum Neural Network Regressor:")
print("  MSE:", mse)
print("  R²:", r2)

sse = np.sum((y_test - y_pred) ** 2)
sst = np.sum((y_test - np.mean(y_test)) ** 2)
print(f"SSE: {sse}, SST: {sst}, R²: {1 - (sse / sst)}")

print("Actual:", y_test[:10])
print("Predicted:", y_pred[:10])
