import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from ipywidgets import interact
from typing import Callable, List, Tuple

FONT_SIZE_TICKS = 15
FONT_SIZE_TITLE = 25
FONT_SIZE_AXES = 20

def normalize_data(
        train_data: pd.core.frame.DataFrame,
        val_data: pd.core.frame.DataFrame,
        test_data: pd.core.frame.DataFrame,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, pd.core.series.Series, pd.core.series.Series
]:
    """Normalizes train, val and test splits.

    Args:
        train_data (pd.core.frame.DataFrame): Train split.
        val_data (pd.core.frame.DataFrame): Validation split.
        test_data (pd.core.frame.DataFrame): Test split.

    Returns:
        tuple: Normalized splits with training mean and standard deviation.
    """
    train_mean = train_data.mean()
    train_std = train_data.std()

    train_data = (train_data - train_mean) / train_std
    val_data = (val_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    return train_data, val_data, test_data, train_mean, train_std


@dataclass
class DataSplits:
    """Class to encapsulate normalized/unnormalized train, val, test, splits."""

    train_data: pd.core.frame.DataFrame
    val_data: pd.core.frame.DataFrame
    test_data: pd.core.frame.DataFrame
    train_mean: pd.core.series.Series
    train_std: pd.core.series.Series
    train_df_unnormalized: pd.core.frame.DataFrame
    val_df_unnormalized: pd.core.frame.DataFrame
    test_df_unnormalized: pd.core.frame.DataFrame


def train_val_test_split(df: pd.core.frame.DataFrame) -> DataSplits:
    """Splits a dataframe into train, val and test.

    Args:
        df (pd.core.frame.DataFrame): The data to split.

    Returns:
        data_splits (DataSplits): An instance that encapsulates normalized/unnormalized splits.
    """
    n = len(df)
    train_df = df[0: int(n * 0.7)]
    val_df = df[int(n * 0.7): int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    train_df_un = train_df.copy(deep=True)
    val_df_un = val_df.copy(deep=True)
    test_df_un = test_df.copy(deep=True)

    train_df_un = train_df_un.mask(train_df_un.WS == -1, np.nan)
    val_df_un = val_df_un.mask(val_df_un.WS == -1, np.nan)
    test_df_un = test_df_un.mask(test_df_un.WS == -1, np.nan)

    train_df, val_df, test_df, train_mn, train_st = normalize_data(
        train_df, val_df, test_df
    )

    ds = DataSplits(
        train_data=train_df,
        val_data=val_df,
        test_data=test_df,
        train_mean=train_mn,
        train_std=train_st,
        train_df_unnormalized=train_df_un,
        val_df_unnormalized=val_df_un,
        test_df_unnormalized=test_df_un,
    )

    return ds


def plot_time_series(data_splits: DataSplits) -> None:
    """Plots time series of active power vs the other features.

    Args:
        data_splits (DataSplits): Turbine data.
    """
    train_df, val_df, test_df = (
        data_splits.train_df_unnormalized,
        data_splits.val_df_unnormalized,
        data_splits.test_df_unnormalized,
    )

    def plot_time_series(feature):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
        ax1.plot(train_df["WS"], color="blue", label="training")
        ax1.plot(val_df["WS"], color="green", label="validation")
        ax1.plot(test_df["WS"], color="red", label="test")
        ax1.set_title("Time series of WS (target)", fontsize=FONT_SIZE_TITLE)
        ax1.set_ylabel("Active Power (m/s)", fontsize=FONT_SIZE_AXES)
        ax1.set_xlabel("Date", fontsize=FONT_SIZE_AXES)
        ax1.legend(fontsize=15)
        ax1.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        ax2.plot(train_df[feature], color="blue", label="training")
        ax2.plot(val_df[feature], color="green", label="validation")
        ax2.plot(test_df[feature], color="red", label="test")
        ax2.set_title(f"Time series of {feature} (predictor)", fontsize=FONT_SIZE_TITLE)
        ax2.set_ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        ax2.set_xlabel("Date", fontsize=FONT_SIZE_AXES)
        ax2.legend(fontsize=15)
        ax2.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        plt.tight_layout()
        plt.show()

    feature_selection = widgets.Dropdown(
        options=[f for f in list(train_df.columns) if f != "WS"],
        description="Feature",
    )

    interact(plot_time_series, feature=feature_selection)


def compute_metrics(
        true_series: np.ndarray, forecast: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes MSE and MAE between two time series.

    Args:
        true_series (np.ndarray): True values.
        forecast (np.ndarray): Forecasts.

    Returns:
        tuple: MSE and MAE metrics.
    """

    mse = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
    mae = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()

    return mse, mae


class WindowGenerator:
    """A utility class for generating windows of time-series data suitable for
    training machine learning models. Provides functionalities for plotting and
    visualizing data windows and predictions."""
    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df, label_columns=["WS"]):
        # Store raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # List of columns to be used as labels for prediction
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def plot(self, model=None, plot_col="WS", max_subplots=1):
        """Visualize the window of input data and labels.

        If a model is provided, it plots its predictions as well.

        Args:
            model (tf.keras.Model, optional): Model to make predictions.
            plot_col (str, optional): Column name to plot. Defaults to "WS".
            max_subplots (int, optional): Maximum number of subplots to display. Defaults to 1.
        """
        inputs, labels = self.example
        plt.figure(figsize=(20, 6))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.title("Inputs (past) vs Labels (future predictions)", fontsize=FONT_SIZE_TITLE)
            plt.ylabel(f"{plot_col} (normalized)", fontsize=FONT_SIZE_AXES)
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                color="green",
                linestyle="--",
                label="Inputs",
                marker="o",
                markersize=10,
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.plot(
                self.label_indices,
                labels[n, :, label_col_index],
                color="orange",
                linestyle="--",
                label="Labels",
                markersize=10,
                marker="o"
            )
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="*",
                    edgecolors="k",
                    label="Predictions",
                    c="pink",
                    s=64,
                )
            plt.legend(fontsize=FONT_SIZE_TICKS)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.xlabel("Timestep", fontsize=FONT_SIZE_AXES)

    def plot_long(
            self,
            model,
            data_splits,
            plot_col="WS",
            time_steps_future=1,
            baseline_mae=None,
    ):
        """Plot long term predictions against real values for the test split.

        Args:
            model (tf.keras.Model): Model to make predictions.
            data_splits: Object containing training mean and standard deviation.
            plot_col (str, optional): Column name to plot. Defaults to "WS".
            time_steps_future (int, optional): Number of time steps to look into the future.
            baseline_mae (float, optional): Baseline mean absolute error for comparison.
        """
        train_mean, train_std = data_splits.train_mean, data_splits.train_std
        self.test_size = len(self.test_df)
        self.test_data = self.make_test_dataset(self.test_df, self.test_size)

        inputs, labels = next(iter(self.test_data))

        plt.figure(figsize=(20, 6))
        plot_col_index = self.column_indices[plot_col]

        plt.ylabel(f"{plot_col} (m/s)", fontsize=FONT_SIZE_AXES)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        labels = (labels * train_std.WS) + train_mean.WS

        upper = 24 - (time_steps_future - 1)
        lower = self.label_indices[-1] - upper
        self.label_indices_long = self.test_df.index[lower:-upper]

        plt.plot(
            self.label_indices_long[:],
            labels[:, time_steps_future - 1, label_col_index][:],
            label="Labels",
            c="Blue",
        )

        if model is not None:
            predictions = model(inputs)
            predictions = (predictions * train_std.WS) + train_mean.WS
            predictions_for_timestep = predictions[
                                       :, time_steps_future - 1, label_col_index
                                       ][:]
            predictions_for_timestep = tf.nn.relu(predictions_for_timestep).numpy()
            plt.plot(
                self.label_indices_long[:],
                predictions_for_timestep,
                label="Predictions",
                c="red",
                linewidth=3,
            )
            plt.legend(fontsize=FONT_SIZE_TICKS)
            _, mae = compute_metrics(
                labels[:, time_steps_future - 1, label_col_index][:],
                predictions_for_timestep,
            )

            if baseline_mae is None:
                baseline_mae = mae

            print(
                f"\nMean Absolute Error (M/S): {mae:.2f} for forecast.\n\nImprovement over Moving Average baseline MAE: {100 * ((baseline_mae - mae) / baseline_mae):.2f}%"
            )
        plt.title("Model Predictions vs Real Values for Test Set Data     ", fontsize=FONT_SIZE_TITLE)
        plt.xlabel("Date", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        return mae

    def make_test_dataset(self, data, bs):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=bs,
        )

        ds = ds.map(self.split_window)

        return ds

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        result = getattr(self, "_example", None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result


def generate_window(
        train_df: pd.core.frame.DataFrame,
        val_df: pd.core.frame.DataFrame,
        test_df: pd.core.frame.DataFrame,
        days_in_past: int,
        width: int = 24
) -> WindowGenerator:
    """Creates a windowed dataset given the train, val, test splits and the number of days into the past.

    Args:
        train_df (pd.core.frame.DataFrame): Train split.
        val_df (pd.core.frame.DataFrame): Val Split.
        test_df (pd.core.frame.DataFrame): Test split.
        days_in_past (int): How many days into the past will be used to predict the next 24 hours.

    Returns:
        WindowGenerator: The windowed dataset.
    """
    OUT_STEPS = 24
    multi_window = WindowGenerator(
        input_width=width * days_in_past,
        label_width=OUT_STEPS,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        shift=OUT_STEPS,
    )
    return multi_window


def create_model(num_features: int, days_in_past: int) -> tf.keras.Model:
    """Creates a Conv-LSTM model for time series prediction.

    Args:
        num_features (int): Number of features used for prediction.
        days_in_past (int): Number of days into the past to predict next 24 hours.

    Returns:
        tf.keras.Model: The uncompiled model.
    """
    CONV_WIDTH = 3
    OUT_STEPS = 24
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Masking(
                mask_value=-1.0, input_shape=(days_in_past * 24, num_features)
            ),
            tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
            tf.keras.layers.Conv1D(256, activation="relu", kernel_size=(CONV_WIDTH)),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True)
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=False)
            ),
            tf.keras.layers.Dense(
                OUT_STEPS * 1, kernel_initializer=tf.initializers.zeros()
            ),
            tf.keras.layers.Reshape([OUT_STEPS, 1]),
        ]
    )

    return model


def compile_and_fit(
        model: tf.keras.Model, window: WindowGenerator, patience: int = 3, Epochs: int = 100
) -> tf.keras.callbacks.History:
    """Compiles and trains a model given a patience threshold.

    Args:
        model (tf.keras.Model): The model to train.
        window (WindowGenerator): The windowed data.
        patience (int, optional): Patience threshold to stop training. Defaults to 2.

    Returns:
        tf.keras.callbacks.History: The training history.
    """
    EPOCHS = Epochs

    # Set up early stopping based on validation loss
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min"  # "min" indicates that training will stop when the monitored quantity stops decreasing
    )
    # Create a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
    'model_checkpoint.h5',
    save_weights_only=False,
    save_best_only=True,
    save_freq='epoch',
    verbose=1
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
    )

    tf.random.set_seed(432)
    np.random.seed(432)
    random.seed(432)

    # Include the checkpoint_callback in the callbacks list
    history = model.fit(
        window.train, epochs=EPOCHS, validation_data=window.val,
        callbacks=[early_stopping, checkpoint_callback]  # Add checkpoint_callback here
    )

    if len(history.epoch) < EPOCHS:
        print(
            "\nTraining stopped early to prevent overfitting, as the validation loss is increasing for two consecutive steps.")

    return history


def train_conv_lstm_model(
        data: pd.core.frame.DataFrame, features: List[str], days_in_past: int
) -> Tuple[WindowGenerator, tf.keras.Model, DataSplits]:
    """Trains the Conv-LSTM model for time series prediction.

    Args:
        data (pd.core.frame.DataFrame): The dataframe to be used.
        data (list[str]): The features to use for forecasting.
        days_in_past (int): How many days in the past to use to forecast the next 24 hours.

    Returns:
        tuple: The windowed dataset, the model that handles the forecasting logic and the data used.
    """
    data_splits = train_val_test_split(data[features])

    train_data, val_data, test_data, train_mean, train_std = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
        data_splits.train_mean,
        data_splits.train_std,
    )

    window = generate_window(train_data, val_data, test_data, days_in_past)
    num_features = window.train_df.shape[1]

    model = create_model(num_features, days_in_past)
    history = compile_and_fit(model, window)

    return window, model, data_splits


def prediction_plot(
        func: Callable, model: tf.keras.Model, data_splits: DataSplits, baseline_mae: float
, prediction = 1) -> None:
    """Plot an interactive visualization of predictions vs true values.

    Args:
        func (Callable): Function to close over. Should be the plot_long method from the WindowGenerator instance.
        model (tf.keras.Model): The trained model.
        data_splits (DataSplits): The data used.
        baseline_mae (float): MAE of baseline to compare against.
    """

    def _plot(time_steps_future):
        mae = func(
            model,
            data_splits,
            time_steps_future=time_steps_future,
            baseline_mae=baseline_mae,
        )

    time_steps_future_selection = widgets.IntSlider(
        value=prediction,
        min=1,
        max=24,
        step=1,
        description="Hours into future",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, time_steps_future=time_steps_future_selection)
