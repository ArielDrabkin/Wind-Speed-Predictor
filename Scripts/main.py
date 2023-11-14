import IMS_API_data
import WS
import json
import pandas as pd
import tensorflow as tf
import os


def get_data_from_api(token_path, station, start_data, end_date):
    # Get data from API
    with open(token_path, 'r') as file:
        TOKEN = json.load(file)
    STATION = station
    START_DATE = start_data
    END_DATE = end_date
    url = f"https://api.ims.gov.il/v1/envista/stations/{STATION}/data?from={START_DATE}&to={END_DATE}"
    data = IMS_API_data.get_met_data(TOKEN, url)
    print("Data fetched successfully")
    df = IMS_API_data.data_to_df(data)
    print("Data converted to DataFrame successfully")
    df = df[["datetime", "WS"]]
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df.set_index('datetime', inplace=True)
    data_splits = WS.train_val_test_split(df)
    print("Data was split successfully")
    return data_splits


def moving_average(data_splits, n_days):
    time_horizon = 2
    # Compute the rolling average
    moving_avg = data_splits.test_df_unnormalized['WS'].rolling(window=n_days * 24).mean()

    # Shift the values to get a prediction for a future time step
    predicted_WS = moving_avg.shift(-time_horizon)
    # Calculate the absolute error
    abs_error = abs(data_splits.test_df_unnormalized['WS'] - moving_avg)
    mae = abs_error.mean()

    print(f"Mean Absolute Error (MAE): {mae:.2f} for forecast")
    return mae


def train_model(data_splits, n_days):
    window = WS.generate_window(data_splits.train_data, data_splits.val_data, data_splits.test_data, n_days)
    num_features = window.train_df.shape[1]
    model = WS.create_model(num_features, n_days)
    return window, model


def run_model(data_splits, n_days, mae, model_path=None):
    if model_path is None:
        window = WS.generate_window(data_splits.train_data, data_splits.val_data, data_splits.test_data, n_days)
        num_features = window.train_df.shape[1]
        model = WS.create_model(num_features, n_days)
        WS.compile_and_fit(model, window)
        WS.prediction_plot(window.plot_long, model, data_splits, baseline_mae=mae, prediction=24)
    else:
        model = tf.keras.models.load_model(model_path)
        window = WS.generate_window(data_splits.train_data, data_splits.val_data, data_splits.test_data, n_days)
        WS.prediction_plot(window.plot_long, model, data_splits, baseline_mae=mae, prediction=24)
    return window, model


if __name__ == '__main__':
    os.chdir('H:\My Drive\my computer\Data Science\Wind predictor')
    token_path = 'Token.json'
    station = 28
    start_data = '2022/10/13'
    end_date = '2023/10/14'
    model_path = 'Models/model_checkpoint.h5'
    n_days = 3
    data_splits = get_data_from_api(token_path, station, start_data, end_date)
    df = pd.read_csv('Data/IMS_data.csv')
    df = df[["datetime", "WS"]]
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df.set_index('datetime', inplace=True)
    data_splits = WS.train_val_test_split(df)
    mae = moving_average(data_splits, n_days)
    window, model = run_model(data_splits, n_days, mae)
