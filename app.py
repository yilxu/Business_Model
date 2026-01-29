import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
#old code#from tensorflow.keras.models import Sequential
#old code#from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, Dropout
# New corrected lines
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, LSTM, Dropout

import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Detect whether Amazon Chronos is available in the environment.
# Chronos requires Python >= 3.10 for recent releases; the local venv
# may be older (e.g. 3.9), so we detect availability and set a flag.
CHRONOS_AVAILABLE = False
CHRONOS_IMPORT_ERROR = None
try:s
    import importlib.util
    if importlib.util.find_spec('chronos') is not None:
        from chronos import BaseChronosPipeline  # type: ignore
        CHRONOS_AVAILABLE = True
except Exception as e:
    CHRONOS_AVAILABLE = False
    CHRONOS_IMPORT_ERROR = e

RESULTS_DIR = "results"

st.set_page_config(page_title="Stock Returns Prediction", layout="wide")
st.title("Stock Returns Prediction Models")
st.markdown("Predicting normalized returns using 1D CNN, Random Forest, LSTM, and Amazon Chronos models")

@st.cache_data
def load_data():
    """Load Excel data"""
    df = pd.read_excel("attached_assets/msft_data_with_daily_returns_copy_2_1767388706480.xlsx")
    return df

def preprocess_data(df):
    """Preprocess data - drop specified columns"""
    original_cols = df.columns.tolist()
    
    cols_to_drop = ['open', 'high', 'low', 'adj close', 'adj high', 'adj open', 'adj low']
    
    cols_to_remove = []
    for col in df.columns:
        if col.lower() in cols_to_drop:
            cols_to_remove.append(col)
    
    df_clean = df.drop(columns=cols_to_remove, errors='ignore')
    
    date_col = None
    for col in df_clean.columns:
        if 'date' in col.lower():
            date_col = col
            break
    
    if date_col:
        dates = df_clean[date_col]
        df_clean = df_clean.drop(columns=[date_col])
    else:
        dates = None
    
    df_numeric = df_clean.select_dtypes(include=[np.number])
    df_numeric = df_numeric.dropna()
    
    return df_numeric, original_cols, cols_to_remove, dates

def split_data_raw(df, train_ratio=0.7, test_ratio=0.2):
    """Split raw data into train, test, and validation sets BEFORE normalization"""
    n = len(df)
    train_end = int(n * train_ratio)
    test_end = int(n * (train_ratio + test_ratio))
    
    train_data = df.iloc[:train_end].copy()
    test_data = df.iloc[train_end:test_end].copy()
    val_data = df.iloc[test_end:].copy()
    
    return train_data, test_data, val_data

def normalize_splits(train_data, test_data, val_data):
    """Normalize data - fit scaler ONLY on training data to prevent leakage"""
    scaler = StandardScaler()
    
    train_normalized = pd.DataFrame(
        scaler.fit_transform(train_data),
        columns=train_data.columns,
        index=train_data.index
    )
    
    test_normalized = pd.DataFrame(
        scaler.transform(test_data),
        columns=test_data.columns,
        index=test_data.index
    )
    
    val_normalized = pd.DataFrame(
        scaler.transform(val_data),
        columns=val_data.columns,
        index=val_data.index
    )
    
    return train_normalized, test_normalized, val_normalized, scaler

def prepare_features_target(df):
    """Separate features and target"""
    returns_col = None
    for col in df.columns:
        if 'return' in col.lower():
            returns_col = col
            break
    
    if returns_col is None:
        returns_col = df.columns[-1]
    
    X = df.drop(columns=[returns_col])
    y = df[returns_col]
    return X, y, returns_col

def create_sequences(X, y, seq_length=10):
    """Create sequences for time series models"""
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X.iloc[i:(i + seq_length)].values)
        ys.append(y.iloc[i + seq_length])
    return np.array(Xs), np.array(ys)

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2 Score': r2}

def train_random_forest(X_train, y_train, X_test, y_test, X_val, y_val):
    """Train Random Forest model - returns predictions for plotting"""
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    val_pred = model.predict(X_val)
    
    train_metrics = calculate_metrics(y_train, train_pred)
    test_metrics = calculate_metrics(y_test, test_pred)
    val_metrics = calculate_metrics(y_val, val_pred)
    
    predictions = {
        'test': {'actual': np.array(y_test), 'predicted': test_pred},
        'val': {'actual': np.array(y_val), 'predicted': val_pred}
    }
    
    return model, train_metrics, test_metrics, val_metrics, predictions

def train_cnn_1d(X_train, y_train, X_test, y_test, X_val, y_val, seq_length=10, epochs=50):
    """Train 1D CNN model - returns predictions for plotting"""
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    
    if len(X_train_seq) == 0 or len(X_test_seq) == 0 or len(X_val_seq) == 0:
        return None, None, None, None, None
    
    n_features = X_train_seq.shape[2]
    
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, n_features)),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=32, 
              validation_data=(X_val_seq, y_val_seq), verbose=0)
    
    train_pred = model.predict(X_train_seq, verbose=0).flatten()
    test_pred = model.predict(X_test_seq, verbose=0).flatten()
    val_pred = model.predict(X_val_seq, verbose=0).flatten()
    
    train_metrics = calculate_metrics(y_train_seq, train_pred)
    test_metrics = calculate_metrics(y_test_seq, test_pred)
    val_metrics = calculate_metrics(y_val_seq, val_pred)
    
    predictions = {
        'test': {'actual': y_test_seq, 'predicted': test_pred},
        'val': {'actual': y_val_seq, 'predicted': val_pred}
    }
    
    return model, train_metrics, test_metrics, val_metrics, predictions

def train_lstm(X_train, y_train, X_test, y_test, X_val, y_val, seq_length=10, epochs=50):
    """Train LSTM model - returns predictions for plotting"""
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    
    if len(X_train_seq) == 0 or len(X_test_seq) == 0 or len(X_val_seq) == 0:
        return None, None, None, None, None
    
    n_features = X_train_seq.shape[2]
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=32, 
              validation_data=(X_val_seq, y_val_seq), verbose=0)
    
    train_pred = model.predict(X_train_seq, verbose=0).flatten()
    test_pred = model.predict(X_test_seq, verbose=0).flatten()
    val_pred = model.predict(X_val_seq, verbose=0).flatten()
    
    train_metrics = calculate_metrics(y_train_seq, train_pred)
    test_metrics = calculate_metrics(y_test_seq, test_pred)
    val_metrics = calculate_metrics(y_val_seq, val_pred)
    
    predictions = {
        'test': {'actual': y_test_seq, 'predicted': test_pred},
        'val': {'actual': y_val_seq, 'predicted': val_pred}
    }
    
    return model, train_metrics, test_metrics, val_metrics, predictions

def train_lstm_3layer(X_train, y_train, X_test, y_test, X_val, y_val, seq_length=10, epochs=50):
    """Train 3-Layer LSTM model with intermittent dropout - returns predictions for plotting"""
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    
    if len(X_train_seq) == 0 or len(X_test_seq) == 0 or len(X_val_seq) == 0:
        return None, None, None, None, None
    
    n_features = X_train_seq.shape[2]
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=32, 
              validation_data=(X_val_seq, y_val_seq), verbose=0)
    
    train_pred = model.predict(X_train_seq, verbose=0).flatten()
    test_pred = model.predict(X_test_seq, verbose=0).flatten()
    val_pred = model.predict(X_val_seq, verbose=0).flatten()
    
    train_metrics = calculate_metrics(y_train_seq, train_pred)
    test_metrics = calculate_metrics(y_test_seq, test_pred)
    val_metrics = calculate_metrics(y_val_seq, val_pred)
    
    predictions = {
        'test': {'actual': y_test_seq, 'predicted': test_pred},
        'val': {'actual': y_val_seq, 'predicted': val_pred}
    }
    
    return model, train_metrics, test_metrics, val_metrics, predictions

def train_chronos(y_train, y_test, y_val, prediction_length=None):
    """Train Amazon Chronos model for time series forecasting - returns predictions for plotting"""
    if not CHRONOS_AVAILABLE:
        msg = (
            f"Chronos is not available in the current environment. "
            f"Install a compatible release (requires Python >= 3.10) or run without Chronos. "
            f"Original import error: {CHRONOS_IMPORT_ERROR!r}"
        )
        raise ImportError(msg)
    
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    
    y_train_values = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
    y_test_values = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    y_val_values = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
    
    context = torch.tensor(y_train_values, dtype=torch.float32)
    
    test_len = len(y_test_values)
    test_forecast = pipeline.predict(context, prediction_length=test_len)
    
    if hasattr(test_forecast, 'numpy'):
        test_pred = test_forecast.numpy()
    else:
        test_pred = np.array(test_forecast)
    
    if len(test_pred.shape) > 1:
        test_pred = np.median(test_pred, axis=1) if test_pred.shape[0] > 1 else test_pred.flatten()
    
    if len(test_pred) > test_len:
        test_pred = test_pred[:test_len]
    elif len(test_pred) < test_len:
        test_pred = np.pad(test_pred, (0, test_len - len(test_pred)), mode='edge')
    
    full_context = torch.tensor(
        np.concatenate([y_train_values, y_test_values]), 
        dtype=torch.float32
    )
    val_len = len(y_val_values)
    val_forecast = pipeline.predict(full_context, prediction_length=val_len)
    
    if hasattr(val_forecast, 'numpy'):
        val_pred = val_forecast.numpy()
    else:
        val_pred = np.array(val_forecast)
    
    if len(val_pred.shape) > 1:
        val_pred = np.median(val_pred, axis=1) if val_pred.shape[0] > 1 else val_pred.flatten()
    
    if len(val_pred) > val_len:
        val_pred = val_pred[:val_len]
    elif len(val_pred) < val_len:
        val_pred = np.pad(val_pred, (0, val_len - len(val_pred)), mode='edge')
    
    train_forecast = pipeline.predict(context[:-10], prediction_length=10)
    if hasattr(train_forecast, 'numpy'):
        train_pred_sample = train_forecast.numpy()
    else:
        train_pred_sample = np.array(train_forecast)
    
    if len(train_pred_sample.shape) > 1:
        train_pred_sample = np.median(train_pred_sample, axis=1) if train_pred_sample.shape[0] > 1 else train_pred_sample.flatten()
    
    train_metrics = calculate_metrics(y_train_values[-10:], train_pred_sample[:10] if len(train_pred_sample) >= 10 else train_pred_sample)
    test_metrics = calculate_metrics(y_test_values, test_pred)
    val_metrics = calculate_metrics(y_val_values, val_pred)
    
    predictions = {
        'test': {'actual': y_test_values, 'predicted': test_pred},
        'val': {'actual': y_val_values, 'predicted': val_pred}
    }
    
    return pipeline, train_metrics, test_metrics, val_metrics, predictions

def plot_predictions(predictions_dict, dataset_name):
    """Create actual vs predicted plots for all models"""
    st.subheader(f"{dataset_name} Set: Actual vs Predicted")
    
    for model_name, preds in predictions_dict.items():
        if preds is None:
            continue
            
        actual = preds['actual']
        predicted = preds['predicted']
        
        plot_df = pd.DataFrame({
            'Index': range(len(actual)),
            'Actual': actual,
            'Predicted': predicted
        })
        
        st.markdown(f"**{model_name}**")
        st.line_chart(plot_df.set_index('Index')[['Actual', 'Predicted']])

def save_results_to_files(results, all_predictions, test_comparison, val_comparison):
    """Save results tables and plots to the results directory"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    tables_dir = os.path.join(session_dir, "tables")
    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    test_comparison.to_csv(os.path.join(tables_dir, "test_set_comparison.csv"))
    val_comparison.to_csv(os.path.join(tables_dir, "validation_set_comparison.csv"))
    
    for model_name, metrics in results.items():
        model_filename = model_name.replace(" ", "_").lower()
        model_df = pd.DataFrame({
            'Dataset': ['Test', 'Validation'],
            'MSE': [metrics['test']['MSE'], metrics['val']['MSE']],
            'RMSE': [metrics['test']['RMSE'], metrics['val']['RMSE']],
            'MAE': [metrics['test']['MAE'], metrics['val']['MAE']],
            'R2 Score': [metrics['test']['R2 Score'], metrics['val']['R2 Score']]
        })
        model_df.to_csv(os.path.join(tables_dir, f"{model_filename}_metrics.csv"), index=False)
    
    for dataset_name, predictions_dict in [('test', all_predictions['test']), ('val', all_predictions['val'])]:
        for model_name, preds in predictions_dict.items():
            if preds is None:
                continue
            
            actual = preds['actual']
            predicted = preds['predicted']
            
            pred_df = pd.DataFrame({
                'Index': range(len(actual)),
                'Actual': actual,
                'Predicted': predicted
            })
            model_filename = model_name.replace(" ", "_").lower()
            pred_df.to_csv(os.path.join(tables_dir, f"{model_filename}_{dataset_name}_predictions.csv"), index=False)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(range(len(actual)), actual, label='Actual', color='blue', alpha=0.7)
            ax.plot(range(len(predicted)), predicted, label='Predicted', color='red', alpha=0.7)
            ax.set_xlabel('Index')
            ax.set_ylabel('Normalized Returns')
            ax.set_title(f'{model_name} - {dataset_name.capitalize()} Set: Actual vs Predicted')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_filename = f"{model_filename}_{dataset_name}_plot.png"
            fig.savefig(os.path.join(plots_dir, plot_filename), dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    return session_dir

# Main App Logic
try:
    st.header("1. Data Loading and Preprocessing")
    
    with st.spinner("Loading data..."):
        df_raw = load_data()
        df_numeric, original_cols, removed_cols, dates = preprocess_data(df_raw)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Columns")
        st.write(original_cols)
    
    with col2:
        st.subheader("Removed Columns")
        st.write(removed_cols if removed_cols else "None")
    
    st.subheader("Kept Columns (Numeric)")
    st.write(df_numeric.columns.tolist())
    
    st.subheader("Data Shape")
    st.write(f"Rows: {df_numeric.shape[0]}, Columns: {df_numeric.shape[1]}")
    
    st.subheader("Data Preview (First 10 rows)")
    st.dataframe(df_numeric.head(10))
    
    st.header("2. Data Splitting (Before Normalization)")
    train_raw, test_raw, val_raw = split_data_raw(df_numeric)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Set (70%)", f"{len(train_raw)} samples")
    with col2:
        st.metric("Test Set (20%)", f"{len(test_raw)} samples")
    with col3:
        st.metric("Validation Set (10%)", f"{len(val_raw)} samples")
    
    st.header("3. Data Normalization (Fit on Training Only)")
    train_data, test_data, val_data, scaler = normalize_splits(train_raw, test_raw, val_raw)
    st.success("Data normalized - Scaler fitted on training data only to prevent data leakage")
    
    st.subheader("Training Data Statistics (After Normalization)")
    st.dataframe(train_data.describe())
    
    X_train, y_train, target_col = prepare_features_target(train_data)
    X_test, y_test, _ = prepare_features_target(test_data)
    X_val, y_val, _ = prepare_features_target(val_data)
    
    st.write(f"**Target column (Returns):** {target_col}")
    st.write(f"**Feature columns:** {X_train.columns.tolist()}")
    
    st.header("4. Model Training and Evaluation")
    
    seq_length = st.slider("Sequence Length (for CNN, LSTM)", 5, 20, 10)
    epochs = st.slider("Training Epochs (for deep learning models)", 10, 100, 50)
    
    if st.button("Train All Models", type="primary"):
        results = {}
        all_predictions = {'test': {}, 'val': {}}
        
        with st.spinner("Training Random Forest..."):
            rf_model, rf_train, rf_test, rf_val, rf_preds = train_random_forest(
                X_train, y_train, X_test, y_test, X_val, y_val
            )
            results['Random Forest'] = {'train': rf_train, 'test': rf_test, 'val': rf_val}
            all_predictions['test']['Random Forest'] = rf_preds['test']
            all_predictions['val']['Random Forest'] = rf_preds['val']
        st.success("Random Forest trained!")
        
        with st.spinner("Training 1D CNN..."):
            cnn_model, cnn_train, cnn_test, cnn_val, cnn_preds = train_cnn_1d(
                X_train, y_train, X_test, y_test, X_val, y_val, seq_length, epochs
            )
            if cnn_train is not None:
                results['1D CNN'] = {'train': cnn_train, 'test': cnn_test, 'val': cnn_val}
                all_predictions['test']['1D CNN'] = cnn_preds['test']
                all_predictions['val']['1D CNN'] = cnn_preds['val']
            else:
                st.warning("1D CNN: Not enough data for the selected sequence length")
        st.success("1D CNN trained!")
        
        with st.spinner("Training LSTM..."):
            lstm_model, lstm_train, lstm_test, lstm_val, lstm_preds = train_lstm(
                X_train, y_train, X_test, y_test, X_val, y_val, seq_length, epochs
            )
            if lstm_train is not None:
                results['LSTM'] = {'train': lstm_train, 'test': lstm_test, 'val': lstm_val}
                all_predictions['test']['LSTM'] = lstm_preds['test']
                all_predictions['val']['LSTM'] = lstm_preds['val']
            else:
                st.warning("LSTM: Not enough data for the selected sequence length")
        st.success("LSTM trained!")
        
        with st.spinner("Training 3-Layer LSTM with Dropout..."):
            lstm3_model, lstm3_train, lstm3_test, lstm3_val, lstm3_preds = train_lstm_3layer(
                X_train, y_train, X_test, y_test, X_val, y_val, seq_length, epochs
            )
            if lstm3_train is not None:
                results['3-Layer LSTM'] = {'train': lstm3_train, 'test': lstm3_test, 'val': lstm3_val}
                all_predictions['test']['3-Layer LSTM'] = lstm3_preds['test']
                all_predictions['val']['3-Layer LSTM'] = lstm3_preds['val']
            else:
                st.warning("3-Layer LSTM: Not enough data for the selected sequence length")
        st.success("3-Layer LSTM trained!")
        
        # Skip Chronos training entirely if chronos is not available.
        if CHRONOS_AVAILABLE:
            with st.spinner("Training Amazon Chronos (downloading model weights...)"):
                try:
                    chronos_model, chronos_train, chronos_test, chronos_val, chronos_preds = train_chronos(
                        y_train, y_test, y_val
                    )
                    results['Amazon Chronos'] = {'train': chronos_train, 'test': chronos_test, 'val': chronos_val}
                    all_predictions['test']['Amazon Chronos'] = chronos_preds['test']
                    all_predictions['val']['Amazon Chronos'] = chronos_preds['val']
                    st.success("Amazon Chronos trained!")
                except Exception as e:
                    st.warning(f"Amazon Chronos failed: {str(e)}")
        else:
            st.info(
                "Amazon Chronos skipped: package not available in this environment. "
                "To enable Chronos, install a compatible Python (>=3.10) and then `pip install chronos-forecasting>=2.2.2`."
            )
        
        st.header("5. Model Results")
        
        for model_name, metrics in results.items():
            st.subheader(f"{model_name}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Test Set Performance**")
                test_df = pd.DataFrame([metrics['test']])
                st.dataframe(test_df.style.format("{:.6f}"))
            
            with col2:
                st.markdown("**Validation Set Performance**")
                val_df = pd.DataFrame([metrics['val']])
                st.dataframe(val_df.style.format("{:.6f}"))
            
            st.markdown("---")
        
        st.header("6. Model Comparison Summary")
        
        st.subheader("Test Set Comparison")
        test_comparison = pd.DataFrame({
            model: metrics['test'] for model, metrics in results.items()
        }).T
        st.dataframe(test_comparison.style.format("{:.6f}").highlight_min(axis=0, subset=['MSE', 'RMSE', 'MAE']).highlight_max(axis=0, subset=['R2 Score']))
        
        st.subheader("Validation Set Comparison")
        val_comparison = pd.DataFrame({
            model: metrics['val'] for model, metrics in results.items()
        }).T
        st.dataframe(val_comparison.style.format("{:.6f}").highlight_min(axis=0, subset=['MSE', 'RMSE', 'MAE']).highlight_max(axis=0, subset=['R2 Score']))
        
        st.header("7. Best Model")
        best_test = min(results.items(), key=lambda x: x[1]['test']['MSE'])
        best_val = min(results.items(), key=lambda x: x[1]['val']['MSE'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Best on Test Set:** {best_test[0]} (MSE: {best_test[1]['test']['MSE']:.6f})")
        with col2:
            st.success(f"**Best on Validation Set:** {best_val[0]} (MSE: {best_val[1]['val']['MSE']:.6f})")
        
        st.header("8. Actual vs Predicted Plots")
        
        plot_predictions(all_predictions['test'], "Test")
        plot_predictions(all_predictions['val'], "Validation")
        
        st.header("9. Save Results")
        with st.spinner("Saving results to files..."):
            session_dir = save_results_to_files(results, all_predictions, test_comparison, val_comparison)
        st.success(f"Results saved to: {session_dir}")
        st.info(f"**Tables saved:** CSV files with metrics and predictions\n\n**Plots saved:** PNG images for each model's actual vs predicted values")

except FileNotFoundError:
    st.error("Data file not found. Please ensure the Excel file is in the attached_assets folder.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
