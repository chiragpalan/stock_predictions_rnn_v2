import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Function to fetch table names from a database
def fetch_table_names(db_path):
    conn = sqlite3.connect(db_path)
    tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    conn.close()
    return tables

# Function to load and plot data for a specific table
def load_and_plot_data(actual_db_path, pred_db_path, selected_table):
    # Connect to the databases
    actual_conn = sqlite3.connect(actual_db_path)
    pred_conn = sqlite3.connect(pred_db_path)

    # Load the actual and predicted data
    actual_df = pd.read_sql(f"SELECT * FROM {selected_table} ORDER BY Datetime;", actual_conn)
    pred_df = pd.read_sql(f"SELECT * FROM {selected_table}_predictions ORDER BY Datetime;", pred_conn)

    actual_conn.close()
    pred_conn.close()

    # Convert datetime columns to datetime format
    actual_df['Datetime'] = pd.to_datetime(actual_df['Datetime'], errors='coerce').dt.tz_localize(None)
    pred_df['Datetime'] = pd.to_datetime(pred_df['Datetime'], errors='coerce').dt.tz_localize(None)

    # Drop duplicate entries in the 'Datetime' column by keeping the last occurrence
    actual_df = actual_df.drop_duplicates(subset=['Datetime'], keep='last')
    pred_df = pred_df.drop_duplicates(subset=['Datetime'], keep='last')

    # Filter data to only include stock market open hours (9:15 AM to 3:30 PM IST)
    market_hours = (pred_df['Datetime'].dt.time >= datetime.strptime('09:15', '%H:%M').time()) & \
                   (pred_df['Datetime'].dt.time <= datetime.strptime('15:30', '%H:%M').time())
    pred_df = pred_df[market_hours]

    # Perform a left join of predictions (base) with actuals
    merged_df = pd.merge(
        pred_df,
        actual_df,
        on='Datetime',
        how='left',
        suffixes=('_pred', '')  # No suffix for actual data
    )

    # Plot the candlestick chart using Plotly
    fig = go.Figure()

    # Add actual data to the chart (if available)
    if not merged_df[['Open', 'High', 'Low', 'Close']].isnull().all().all():
        fig.add_trace(go.Candlestick(
            x=merged_df['Datetime'],
            open=merged_df['Open'],
            high=merged_df['High'],
            low=merged_df['Low'],
            close=merged_df['Close'],
            name='Actual Data',
            increasing_line_color='green',
            decreasing_line_color='red',
        ))

    # Add predicted data to the chart
    fig.add_trace(go.Candlestick(
        x=merged_df['Datetime'],
        open=merged_df['Predicted_Open'],
        high=merged_df['Predicted_High'],
        low=merged_df['Predicted_Low'],
        close=merged_df['Predicted_Close'],
        name='Predicted Data',
        increasing_line_color='blue',
        decreasing_line_color='orange',
    ))

    # Update layout for better visuals
    fig.update_layout(
        title=f"Candlestick Chart for {selected_table}",
        xaxis_title="Datetime",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            tickformat='%Y-%m-%d %H:%M',
            tickmode='auto',
            showgrid=True,
            type='date'
        ),
        yaxis=dict(
            showgrid=True
        ),
        width=1200,
        height=600
    )

    return fig

# Streamlit App
def main():
    st.title("Stock Data Viewer with Predictions")

    # Database paths
    actual_db_path = 'stock_datamanagement/nifty50_data_v1.db'
    pred_db_path = 'stock_datamanagement/predictions/predictions.db'

    # Fetch table names
    actual_tables = fetch_table_names(actual_db_path)
    pred_tables = fetch_table_names(pred_db_path)

    # Combine actual and predicted table names for selection
    table_options = list(set(actual_tables) & set([t.replace('_predictions', '') for t in pred_tables]))

    # Table selection
    selected_table = st.selectbox("Select a table:", table_options)

    if selected_table:
        st.write(f"Displaying data for: {selected_table}")
        # Load and plot data
        fig = load_and_plot_data(actual_db_path, pred_db_path, selected_table)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
