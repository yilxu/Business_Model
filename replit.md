# Stock Returns Prediction App

## Overview
A Streamlit application that predicts normalized stock returns using 4 different machine learning models: 1D CNN, Random Forest, LSTM, and Amazon Chronos.

## Features
- Loads MSFT stock data from Excel file
- Drops specified columns (Open, High, Low, Adj Close, Adj High, Adj Open, Adj Low)
- Keeps Close and returns columns
- Normalizes/standardizes data (scaler fitted on training data only to prevent data leakage)
- Splits data into Train (70%), Test (20%), Validation (10%)
- Trains and evaluates 4 models:
  - Random Forest
  - 1D CNN
  - LSTM
  - Amazon Chronos (pre-trained time series model)
- Displays MSE, RMSE, MAE, R2 Score for test and validation sets

## How to Use
1. The app automatically loads the data file from `attached_assets/`
2. Review the data preprocessing and splitting information
3. Adjust sequence length and epochs using the sliders
4. Click "Train All Models" to train all 4 models
5. View results in comparison tables

## Technical Details
- Framework: Streamlit
- ML Libraries: scikit-learn, TensorFlow/Keras, PyTorch
- Time Series Model: Amazon Chronos (chronos-forecasting)
- Port: 5000

### Chronos / Python compatibility
Recent `chronos-forecasting` releases (>=2.x) require Python >= 3.10. If your environment uses Python 3.9 (the project venv here is 3.9), install a Chronos release compatible with Python 3.9, for example:

```bash
pip install "chronos-forecasting==1.5.3"
```

If you want the newer Chronos features, upgrade your interpreter to Python >= 3.10, recreate the virtualenv, and then install a 2.x Chronos release.

## Running the App
```
streamlit run app.py --server.port 5000
```
