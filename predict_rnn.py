import os
import sqlite3
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Function to load data for prediction
def load_data_for_prediction(table_name, conn, time_steps=30):
    df = pd.read_sql(f"SELECT * FROM {table_name};", conn)
    if 'Datetime' not in df.columns:
        raise KeyError(f"'Datetime' column not found in table {table_name}")
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df.dropna(subset=['Datetime'], inplace=True)
    df.drop_duplicates(subset=['Datetime'], inplace=True)
    df.sort_values('Datetime', inplace=True)
    return df

# Function to preprocess data for prediction
def preprocess_data_for_prediction(df, scaler, time_steps=30):
    input_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[input_columns] = scaler.transform(df[input_columns])
    X = []
    for i in range(len(df) - time_steps):
        X.append(df[input_columns].iloc[i:i + time_steps].values)
    return np.array(X)

# Function to make predictions
def make_predictions(model, X, scaler):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions.reshape(-1, predictions.shape[2])).reshape(predictions.shape)
    return predictions

# Function to generate valid stock market open timestamps
def generate_valid_timestamps(start_datetime, num_predictions=5):
    timestamps = []
    current_datetime = start_datetime
    if current_datetime.time() >= pd.Timestamp('15:25').time():
        current_datetime += pd.Timedelta(days=1)
        current_datetime = current_datetime.replace(hour=9, minute=15)
    while len(timestamps) < num_predictions:
        if current_datetime.weekday() < 5 and current_datetime.time() >= pd.Timestamp('09:15').time() and current_datetime.time() <= pd.Timestamp('15:30').time():
            timestamps.append(current_datetime)
        current_datetime += pd.Timedelta(minutes=5)
        if current_datetime.time() > pd.Timestamp('15:30').time():
            current_datetime += pd.Timedelta(days=1)
            current_datetime = current_datetime.replace(hour=9, minute=15)
    return timestamps

# Function to store predictions
def store_predictions(predictions, table_name, timestamps, db_name="predictions.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Check if the table already exists
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    table_exists = cursor.fetchone() is not None
    
    df_predictions = pd.DataFrame({
        'Datetime': timestamps,
        'Predicted_Open': predictions[:, 0, 0],
        'Predicted_High': predictions[:, 0, 1],
        'Predicted_Low': predictions[:, 0, 2],
        'Predicted_Close': predictions[:, 0, 3],
        'Predicted_Volume': predictions[:, 0, 4],
    })
    
    if table_exists:
        # Append the new predictions to the existing table
        df_predictions.to_sql(table_name, conn, if_exists='append', index=False)
    else:
        # Create a new table if it does not exist
        df_predictions.to_sql(table_name, conn, if_exists='replace', index=False)
    
    conn.close()

def main():
    all_stocks_db = "nifty50_data_v1.db"
    predictions_db = "predictions/predictions.db"
    folder_name = "models"
    
    conn = sqlite3.connect(all_stocks_db)
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    table_names = pd.read_sql(tables_query, conn)['name'].tolist()
    table_names = [table_name for table_name in table_names if table_name != 'sqlite_sequence'] 
    
    for table_name in table_names:
        print(f"Predicting for table: {table_name}")
        
        try:
            df = load_data_for_prediction(table_name, conn)
        except KeyError as e:
            print(e)
            continue
        
        model_path = os.path.join(folder_name, f"{table_name}_model.h5")
        scaler_path = os.path.join(folder_name, f"{table_name}_scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Model or scaler for {table_name} not found. Skipping...")
            continue
        
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        X = preprocess_data_for_prediction(df, scaler)
        
        # Select the last 150 instances for prediction
        X_last_150 = X[-30:]
        
        latest_datetime = df['Datetime'].iloc[-1]
        latest_datetime = latest_datetime.replace(tzinfo=None)  # Remove timezone information
        timestamps = generate_valid_timestamps(latest_datetime, num_predictions=30)
        
        predictions = make_predictions(model, X_last_150, scaler)
        
        store_predictions(predictions, f"{table_name}_predictions", timestamps, predictions_db)
    
    conn.close()

if __name__ == "__main__":
    main()
