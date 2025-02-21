
# Battery Life Prediction Hybrid Model

This repository contains Python programs that predict the battery life of various devices using a **hybrid model** approach. The models combine analytical formulas based on domain knowledge with machine learning techniques (both classical and quantum) to provide robust battery life estimates.

## Contents

- **device_battery_data.xlsx**  
  An Excel file with raw battery data. It includes:
  - Battery capacity (mAh) and voltage (V)
  - Charging cycles versus maximum cycles
  - Device age and battery lifespan
  - Active and sleep power consumption details
  - Operating temperature and environmental conditions

- **LinearGradientBoosing.py**  
  A Python script that:
  - **Data Processing & Formula-Based Estimation:**  
    - Computes engineered features such as the Battery Degradation Factor, Average Power Consumption, Battery Capacity in mWh, and adjustment factors based on temperature and environment.
    - Uses a domain-specific formula to initially estimate the battery life:
      
      ```
      Battery_Life = (Battery_Capacity_mWh * Battery_Degradation_Factor * Temperature Factor * Environmental Factor) / Average_Power_Consumption
      ```
      
  - **Machine Learning & Ensemble Approach:**  
    - Splits the data into training and testing sets.
    - Trains two classical regression models:
      - Gradient Boosting Regressor
      - XGBoost Regressor
    - Combines the predictions from both models into an ensemble (hybrid) prediction by averaging the results.
    - Outputs performance metrics (MSE and R²) and a comparison of actual versus predicted battery life values.

- **QisKit.py**  
  A Python script that:
  - **Data Processing & Formula-Based Estimation:**  
    - Performs similar feature engineering as the first script, using formulas to compute a preliminary battery life estimate.
  - **Quantum Machine Learning Approach:**  
    - Standardizes and reduces the data dimensions with PCA.
    - Constructs a Quantum Neural Network (QNN) using Qiskit's tools (ZZFeatureMap and EfficientSU2 circuits) to capture complex patterns.
    - Trains the QNN using a neural network regressor with a quantum optimizer.
    - Evaluates the model with performance metrics (MSE and R²) and displays actual vs. predicted values.

## Hybrid Model Approach

Both programs leverage a **hybrid strategy**:
- **Formula-Based Estimation:**  
  They use physics-inspired formulas to calculate key parameters (such as degradation factors and capacity conversions) that capture domain-specific insights about battery behavior.
- **Machine Learning Refinement:**  
  The initial formula-based estimates are then refined using machine learning models. In the classical script, two ensemble models (Gradient Boosting and XGBoost) are used, while the quantum script employs a Quantum Neural Network to capture non-linear relationships in the data.

This combination allows the models to benefit from both theoretical understanding and data-driven adjustments, leading to more reliable battery life predictions.

## Requirements

Ensure you have the following libraries installed:
- **Common dependencies:** `numpy`, `pandas`, `scikit-learn`
- **For LinearGradientBoosing.py:** `xgboost`
- **For QisKit.py:** `qiskit`, `qiskit-aer`, `qiskit-machine-learning`, `qiskit-algorithms`

You can install the necessary packages using pip, for example:

```bash
pip install numpy pandas scikit-learn xgboost qiskit qiskit-aer qiskit-machine-learning qiskit-algorithms
```

## Usage

1. **Update File Paths:**  
   Both scripts refer to the battery data file at:  
   `C:/Users/SWARNAJIT ROY/Desktop/Projects/BatteryPrediction/Battery_Life_Prediction/device_battery_data.xlsx`  
   Modify the paths in the scripts if your file is located elsewhere.

2. **Run the Scripts:**  
   Execute each script from the command line or your preferred Python IDE:
   - For classical hybrid prediction:
     ```bash
     python LinearGradientBoosing.py
     ```
   - For quantum neural network prediction:
     ```bash
     python QisKit.py
     ```

## Data Processing and Feature Engineering

Both scripts perform the following steps:
- **Battery Degradation Factor:**  
  Calculated using device-specific coefficients (alpha and beta) and the relationship between charging cycles, device age, and battery lifespan.
- **Average Power Consumption:**  
  Derived from active and sleep power consumption over a 24-hour period.
- **Battery Capacity (mWh):**  
  Converts battery capacity from mAh to mWh using battery voltage.
- **Adjustment Factors:**  
  Temperature and environmental conditions are factored in to adjust battery performance.
- **Battery Life Formula:**  
  Combines the above elements to provide an initial estimate:
  
  ```
  Battery_Life = (Battery_Capacity_mWh * Battery_Degradation_Factor * Temperature Factor * Environmental Factor) / Average_Power_Consumption
  ```

## Model Training and Evaluation

- **LinearGradientBoosing.py:**  
  - Trains classical regression models (Gradient Boosting and XGBoost) on the engineered features.
  - Averages the predictions from both models to produce a hybrid ensemble prediction.
  - Outputs detailed performance metrics and a side-by-side comparison of actual vs. predicted values.

- **QisKit.py:**  
  - Uses PCA and standard scaling before training a Quantum Neural Network regressor.
  - Evaluates the quantum model using similar metrics and outputs a comparison of actual vs. predicted battery life.

