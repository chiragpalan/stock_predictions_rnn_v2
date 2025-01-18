import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from datetime import time, timedelta

def get_next_market_time(current_time):
    """
    Calculate the next valid market time.
    If the current time exceeds 3:30 PM, move to the next trading day at 9:15 AM.
    Args:
        current_time (pd.Timestamp): Current time to check.
    Returns:
        pd.Timestamp: Adjusted time within market hours.
    """
    market_start = time(9, 15)  # Market opens at 9:15 AM
    market_end = time(15, 30)  # Market closes at 3:30 PM

    if current_time.time() >= market_end:  # After market close
        # Move to the next trading day at 9:15 AM
        next_day = current_time + pd.Timedelta(days=1)
        return pd.Timestamp(next_day.date()) + pd.Timedelta(hours=9, minutes=15)
    elif current_time.time() < market_start:  # Before market open
        # Adjust to today's market start time
        return pd.Timestamp(current_time.date()) + pd.Timedelta(hours=9, minutes=15)
    return current_time


def preprocess_new_data(df):
    """
    Preprocess new data by converting the Datetime column, removing duplicates,
    and sorting by datetime.
    Args:
        df (pd.DataFrame): The raw DataFrame.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Convert 'Datetime' column to datetime format, removing timezone information
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df['Datetime'] = df['Datetime'].dt.tz_localize(None)  # Remove timezone info
    print("Datetime_test_data", df.tail())

    # Remove rows with invalid or missing datetime values
    df.dropna(subset=['Datetime'], inplace=True)

    # Remove duplicates and sort by datetime
    df.drop_duplicates(subset=['Datetime'], inplace=True)
    df.sort_values('Datetime', inplace=True)

    return df


def create_sequences(data, input_columns, n_steps):
    """
    Create input sequences for RNN model.
    Args:
        data (pd.DataFrame): The data to create sequences from.
        input_columns (list): Columns to be used as input features.
        n_steps (int): Number of timesteps for input sequence.
    Returns:
        np.ndarray: Input sequences for prediction.
    """
    X = []
    for i in range(len(data) - n_steps):
        X.append(data[input_columns].iloc[i:i+n_steps].values)
    return np.array(X)

def save_predictions_to_db(predictions, datetimes, db_path, table_name, scaler):
    """
    Save predictions for multiple future steps, ensuring times are within market hours.
    """
    predictions = scaler.inverse_transform(predictions.reshape(-1, predictions.shape[2])).reshape(predictions.shape)
    conn = sqlite3.connect(db_path)
    rows = []

    for i, datetime in enumerate(datetimes):
        base_datetime = datetime
        for step in range(predictions.shape[1]):
            # Calculate the next prediction time
            next_time = base_datetime + pd.Timedelta(minutes=5 * (step + 1))
            next_time = get_next_market_time(next_time)

            rows.append({
                'Datetime': next_time,
                'Predicted_Open': predictions[i, step, 0],
                'Predicted_High': predictions[i, step, 1],
                'Predicted_Low': predictions[i, step, 2],
                'Predicted_Close': predictions[i, step, 3],
                'Predicted_Volume': predictions[i, step, 4],
            })

    pd.DataFrame(rows).to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def main():
    database_path = 'nifty50_data_v1.db'
    predictions_db_path = 'predictions/predictions.db'
    input_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    n_steps = 12
    n_future = 3

    conn = sqlite3.connect(database_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
    tables.remove('sqlite_sequence')

    for table_name in tables:
        df = pd.read_sql(f"SELECT * FROM {table_name};", conn)
        df = preprocess_new_data(df)

        model_path = os.path.join('models', f'{table_name}_model.h5')
        scaler_path = os.path.join('models', f'{table_name}_scaler.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Model or scaler for {table_name} not found. Skipping...")
            continue

        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        df[input_columns] = scaler.transform(df[input_columns])
        X = create_sequences(df, input_columns, n_steps)

        predictions = model.predict(X)
        # print("predictions",predictions)
        save_predictions_to_db(predictions, df['Datetime'].iloc[n_steps:], predictions_db_path, f'{table_name}_predictions', scaler)
    conn.close()
if __name__ == "__main__":
    main()
