import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from datetime import time, timedelta

def get_next_market_time(current_time):
    market_start = time(9, 15)
    market_end = time(15, 30)
    if current_time.time() >= market_end:
        next_day = current_time + pd.Timedelta(days=1)
        return pd.Timestamp(next_day.date()) + pd.Timedelta(hours=9, minutes=15)
    elif current_time.time() < market_start:
        return pd.Timestamp(current_time.date()) + pd.Timedelta(hours=9, minutes=15)
    return current_time

def preprocess_new_data(df):
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df['Datetime'] = df['Datetime'].dt.tz_localize(None)
    df.dropna(subset=['Datetime'], inplace=True)
    df.drop_duplicates(subset=['Datetime'], inplace=True)
    df.sort_values('Datetime', inplace=True)
    return df

def create_sequences(data, input_columns, n_steps):
    X = []
    for i in range(len(data) - n_steps):
        X.append(data[input_columns].iloc[i:i+n_steps].values)
    return np.array(X)

def save_predictions_to_db(predictions, datetimes, db_path, table_name, scaler):
    predictions = scaler.inverse_transform(predictions.reshape(-1, predictions.shape[2])).reshape(predictions.shape)
    conn = sqlite3.connect(db_path)
    rows = []
    for i, datetime in enumerate(datetimes):
        base_datetime = datetime
        for step in range(predictions.shape[1]):
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
    n_steps = 30
    n_future = 5
    conn = sqlite3.connect(database_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
    tables.remove('sqlite_sequence')
    for table_name in tables:
        df = pd.read_sql(f"SELECT * FROM {table_name};", conn)
        df = preprocess_new_data(df)
        df = df.tail(150)
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
        save_predictions_to_db(predictions, df['Datetime'].iloc[n_steps:], predictions_db_path, f'{table_name}_predictions', scaler)
    conn.close()

if __name__ == "__main__":
    main()
