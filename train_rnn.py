import sqlite3
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.losses import MeanSquaredError
import random

def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Seed value for randomness.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Ensure TensorFlow uses deterministic behavior
    tf.config.experimental.enable_op_determinism = True

# Set the random seed
set_random_seed(42)

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

def preprocess_data(database_path):
    conn = sqlite3.connect(database_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
    tables.remove('sqlite_sequence')    
    data = {}
    for table in tables:
        df = pd.read_sql(f"SELECT * FROM {table};", conn)
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df.dropna(subset=['Datetime'], inplace=True)
        df.drop_duplicates(subset=['Datetime'], inplace=True)
        df.sort_values('Datetime', inplace=True)
        data[table] = df
    conn.close()
    return data

def scale_data(df, input_columns):
    scaler = MinMaxScaler()
    df[input_columns] = scaler.fit_transform(df[input_columns])
    return df, scaler

def split_data(df, train_ratio=0.8):
    n_train = int(len(df) * train_ratio)
    train_data = df[:n_train]
    test_data = df[n_train:]
    return train_data, test_data

def create_sequences(data, input_columns, output_columns, n_steps, n_future):
    """
    Create sequences for RNN model training.

    Args:
        data (pd.DataFrame): Input data.
        input_columns (list): Columns to be used as input features.
        output_columns (list): Columns to be used as output features.
        n_steps (int): Number of timesteps in the input sequence.
        n_future (int): Number of timesteps to predict.

    Returns:
        tuple: Input (X) and target (y) sequences.
    """
    X, y = [], []
    for i in range(len(data) - n_steps - n_future + 1):
        X.append(data[input_columns].iloc[i:i + n_steps].values)
        y.append(data[output_columns].iloc[i + n_steps:i + n_steps + n_future].values)
    return np.array(X), np.array(y)

def train_rnn_model(X_train, y_train):
    """
    Train the RNN model.

    Args:
        X_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training target data.

    Returns:
        Model: Trained RNN model.
    """
    model = Sequential([
        LSTM(128, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        GRU(64, activation='relu', return_sequences=True),
        Dropout(0.2),
        GRU(32, activation='relu'),
        Dense(y_train.shape[1] * y_train.shape[2]),  # Output for all timesteps and features
        tf.keras.layers.Reshape((y_train.shape[1], y_train.shape[2]))  # Reshape to (n_future, output_columns)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    return model

def main():
    database_path = 'nifty50_data_v1.db'
    data = preprocess_data(database_path)

    for table_name, df in data.items():
        input_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        output_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        n_steps = 12
        n_future = 3

        df, scaler = scale_data(df, input_columns)
        train_data, test_data = split_data(df)

        X_train, y_train = create_sequences(train_data, input_columns, output_columns, n_steps, n_future)

        model = train_rnn_model(X_train, y_train)

        model_path = os.path.join('models', f'{table_name}_model.h5')
        scaler_path = os.path.join('models', f'{table_name}_scaler.pkl')

        model.save(model_path)
        joblib.dump(scaler, scaler_path)
if __name__ == "__main__":
    main()
