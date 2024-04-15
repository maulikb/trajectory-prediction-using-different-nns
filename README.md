# Trajectory Prediction for Autonomous Vehicles

## Prerequisites
- MATLAB
- Autonomous Car Data (provided in the project folder)
- MATLAB Deep Learning Toolbox for the LSTM function

## Setup
1. Ensure that MATLAB is installed on your computer.
2. Locate the `autonomous-car-data` directory within the project folder.

## Running the Project

### Data Preprocessing
- Open `Main_Trajectory_Prediction.m` in MATLAB.
- This script will preprocess the data, creating training, testing, and validation datasets.
- It will save these datasets as CSV files for further use.

### Model Training
- After preprocessing the data, run each model's MATLAB file:
  - `lstm_with_tanh.m`
  - Follow with the other four model files.
- These scripts will train the models and create `.mat` files containing the trained models.

### Model Evaluation
- Re-run `Main_Trajectory_Prediction.m` to load the trained models.
- This script will also perform comparison and analysis of RMSE for each model.

### Analysis
- Review the output from `Main_Trajectory_Prediction.m` for the RMSE comparison.
- Use the results to understand model performance and make decisions on model improvements.

Make sure to follow these steps in order to run the project correctly.
