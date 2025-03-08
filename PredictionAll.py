import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_excel('Battery_Life_Prediction.xlsx')

# Feature Engineering
data['Battery_Degradation_Factor'] = (
    1 - (data['Charging_Cycles'] / data['Maximum Cycles']) -
    (data['Device Age (years)'] / data['Battery Lifespan (years)'])
)

data['Average_Power_Consumption'] = (
    (data['Active Time per Day (hours)'] * data['Active Power Consumption (mW)']) +
    (data['Sleep Time per Day (hours)'] * data['Sleep Power Consumption (mW)'])
) / 24

data['Battery_Capacity_mWh'] = data['Battery Capacity (mAh)'] * data['Battery Voltage (V)']

data['Battery_Life'] = (
    data['Battery_Capacity_mWh'] * data['Battery_Degradation_Factor'] / data['Average_Power_Consumption']
)

# Model Training
features = ['Battery_Degradation_Factor', 'Average_Power_Consumption', 'Battery_Capacity_mWh']
target = 'Battery_Life'
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} Model: MSE: {mse:.4f}, R²: {r2:.4f}\n")

# Train and evaluate multiple models
evaluate_model("Linear Regression", LinearRegression(), X_train, X_test, y_train, y_test)
evaluate_model("Decision Tree", DecisionTreeRegressor(random_state=42), X_train, X_test, y_train, y_test)
evaluate_model("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test, y_train, y_test)
evaluate_model("Support Vector Machine", SVR(), X_train, X_test, y_train, y_test)
evaluate_model("k-Nearest Neighbors", KNeighborsRegressor(n_neighbors=5), X_train, X_test, y_train, y_test)
evaluate_model("Gaussian Process", GaussianProcessRegressor(), X_train, X_test, y_train, y_test)

evaluate_model("Gradient Boosting", GradientBoostingRegressor(random_state=42), X_train, X_test, y_train, y_test)
evaluate_model("XGBoost", XGBRegressor(n_estimators=100, random_state=42), X_train, X_test, y_train, y_test)

# Neural Network Model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train, y_train, epochs=100, verbose=0)
y_pred_nn = nn_model.predict(X_test).flatten()
evaluate_model("Neural Network", nn_model, X_train, X_test, y_train, y_pred_nn)
