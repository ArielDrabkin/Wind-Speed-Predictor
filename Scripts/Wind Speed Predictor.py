import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import metrics
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from ipywidgets import interact
from typing import Callable, List, Tuple, Dict, Optional
import pickle



class WindowGenerator:
    """A utility class for generating windows of time-series data suitable for
    training machine learning models. Provides functionalities for plotting and
    visualizing data windows and predictions."""

    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df, label_columns=["Patv"]):
        # Store raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # List of columns to be used as labels for prediction
        self.label_columns = label_columns
        if label_columns:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        # Mapping from column names to their indices
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Parameters for windowing the data
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        # Total size of the window
        self.total_window_size = input_width + shift

        # Indices for the input and label data within the window
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        """Splits a batch of data into input features and labels."""
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        # Ensure the inputs and labels have the right shapes
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col="Patv", max_subplots=1):
        """Visualize the window of input data and labels.

        If a model is provided, it plots its predictions as well.

        Args:
            model (tf.keras.Model, optional): Model to make predictions.
            plot_col (str, optional): Column name to plot. Defaults to "Patv".
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
        plot_col="Patv",
        time_steps_future=1,
        baseline_mae=None,
    ):
        """Plot long term predictions against real values for the test split.

        Args:
            model (tf.keras.Model): Model to make predictions.
            data_splits: Object containing training mean and standard deviation.
            plot_col (str, optional): Column name to plot. Defaults to "Patv".
            time_steps_future (int, optional): Number of time steps to look into the future.
            baseline_mae (float, optional): Baseline mean absolute error for comparison.
        """
        train_mean, train_std = data_splits.train_mean, data_splits.train_std
        self.test_size = len(self.test_df)
        self.test_data = self.make_test_dataset(self.test_df, self.test_size)

        inputs, labels = next(iter(self.test_data))

        plt.figure(figsize=(20, 6))
        plot_col_index = self.column_indices[plot_col]

        plt.ylabel(f"{plot_col} (kW)", fontsize=FONT_SIZE_AXES)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        labels = (labels * train_std.Patv) + train_mean.Patv

        upper = 24 - (time_steps_future - 1)
        lower = self.label_indices[-1] - upper
        self.label_indices_long = self.test_df.index[lower:-upper]

        plt.plot(
            self.label_indices_long[:],
            labels[:, time_steps_future - 1, label_col_index][:],
            label="Labels",
            c="green",
        )

        if model is not None:
            predictions = model(inputs)
            predictions = (predictions * train_std.Patv) + train_mean.Patv
            predictions_for_timestep = predictions[
                :, time_steps_future - 1, label_col_index
            ][:]
            predictions_for_timestep = tf.nn.relu(predictions_for_timestep).numpy()
            plt.plot(
                self.label_indices_long[:],
                predictions_for_timestep,
                label="Predictions",
                c="orange",
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
                f"\nMean Absolute Error (kW): {mae:.2f} for forecast.\n\nImprovement over random baseline: {100*((baseline_mae -mae)/baseline_mae):.2f}%"
            )
        plt.title("Predictions vs Real Values for Test Split", fontsize=FONT_SIZE_TITLE)
        plt.xlabel("Date", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        return mae

    def make_test_dataset(self, data, bs):
        """Prepare a TensorFlow dataset suitable for model evaluation.

        Args:
            data (pd.DataFrame): Data to be windowed.
            bs (int): Batch size.

        Returns:
            tf.data.Dataset: Windowed dataset.
        """
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
        """Prepare a TensorFlow dataset suitable for model training/validation.

        Args:
            data (pd.DataFrame): Data to be windowed.

        Returns:
            tf.data.Dataset: Windowed dataset.
        """
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
        """tf.data.Dataset: Returns a windowed dataset for training."""
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        """tf.data.Dataset: Returns a windowed dataset for validation."""
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        """tf.data.Dataset: Returns a windowed dataset for testing."""
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Tuple of tf.Tensor: Returns an example batch of `inputs, labels` from the training dataset."""
        result = getattr(self, "_example", None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result

def generate_window(train_df, val_df, test_df, days_in_past, width=24):
    """Generate a windowed dataset from train, validation, and test data.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        days_in_past (int): Number of days of past data to consider for predictions.
        width (int, optional): Size of the prediction window. Defaults to 24.

    Returns:
        WindowGenerator: An instance of WindowGenerator configured with the provided data.
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

def create_model(num_features: int, days_in_past: int, out_steps = 24) -> tf.keras.Model:
    """
    Constructs a Conv-LSTM model tailored for time series prediction.

    This model predicts the next 24 hours based on the provided past days of data.
    It leverages a 1D convolution layer to recognize local patterns and an LSTM
    layer to understand long-term dependencies in the time series.

    Parameters:
    - num_features: The number of features (variables) in the dataset.
    - days_in_past: Number of past days the model will consider for predictions.

    Returns:
    - A TensorFlow Keras model ready to be compiled and trained.
    """
    # Constants
    CONV_WIDTH = 3  # Width of the convolution filter.
    OUT_STEPS = out_steps  # Number of prediction steps (next 24 hours).

    # Constructing the Sequential model
    model = tf.keras.Sequential(
        [
            # Masking layer: It ignores any input time step with a value of -1.0.
            # This is useful for sequences of varying lengths or with missing values.
            tf.keras.layers.Masking(mask_value=-1.0, input_shape=(days_in_past * 24, num_features)),

            # Lambda layer: Slices the input sequence to consider only the last 'CONV_WIDTH' time steps.
            # This reduces the sequence length to the most recent data before convolution.
            tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),

            # 1D Convolutional layer: Detects local patterns using a filter of width 'CONV_WIDTH'.
            # It has 256 filters and uses the RELU activation function.
            tf.keras.layers.Conv1D(256, activation="relu", kernel_size=(CONV_WIDTH)),

            # First Bidirectional LSTM layer: Captures long-term dependencies in both directions.
            # It returns sequences, allowing the next LSTM layer to also operate over sequences.
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),

            # Second Bidirectional LSTM layer: Captures more temporal dependencies.
            # This does not return sequences, thus only providing the final output.
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)),

            # Dense layer: A fully connected layer that outputs the predicted values for the next 24 hours.
            # The 'kernel_initializer' ensures the initial weights of this layer are zeros.
            tf.keras.layers.Dense(OUT_STEPS * 1, kernel_initializer=tf.initializers.zeros()),

            # Reshape layer: Reshapes the output into the shape of [OUT_STEPS, 1].
            # This ensures the model's output is 24 sequences of single values.
            tf.keras.layers.Reshape([OUT_STEPS, 1]),
        ]
    )

    return model

def compile_and_fit(
    model: tf.keras.Model, window: WindowGenerator, patience: int = 2, Epochs: int = 20
) -> tf.keras.callbacks.History:
    """
    Compiles, trains, and applies early stopping to a TensorFlow Keras model.

    The function sets deterministic seeds to ensure reproducibility. It uses
    Mean Squared Error as the loss function and the Adam optimizer for training.
    Early stopping is implemented based on the validation loss.

    Parameters:
    - model: The TensorFlow Keras model to be compiled and trained.
    - window: A windowed dataset, which contains both training and validation data.
    - patience (optional): Number of epochs with no improvement in validation loss
      after which training will be stopped. Default is 2.

    Returns:
    - The history of the training process.
    """

    # Define the maximum number of epochs for training
    EPOCHS = Epochs

    # Set up early stopping based on validation loss
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min"  # "min" indicates that training will stop when the monitored quantity stops decreasing
    )

    # Compile the model with Mean Squared Error as the loss and Adam as the optimizer
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
    )

    # Seeds set for reproducibility across different runs
    tf.random.set_seed(432)
    np.random.seed(432)
    random.seed(432)

    # Train the model using the training data, while also validating with the validation data
    history = model.fit(
        window.train,
        epochs=EPOCHS,
        validation_data=window.val,
        callbacks=[early_stopping]  # Applying early stopping during training
    )

    # Notify the user if early stopping was triggered
    if len(history.epoch) < EPOCHS:
        print("\nTraining stopped early to prevent overfitting, as the validation loss didn't improve for {} epochs.".format(patience))

    return history


def train_conv_lstm_model(train_data, val_data, test_data, features, days_in_past):
    """
    Trains a Conv-LSTM model on the provided time series dataset using specified features
    and a defined historical window for forecasting the next 24 hours.

    Parameters:
        data (pd.core.frame.DataFrame): Time series dataset containing the data.
        features (list[str]): List of feature columns to use from the dataset.
        days_in_past (int): Number of past days to use for forecasting the subsequent 24-hour period.

    Returns:
        WindowGenerator: A windowed representation of the data used for training the model.
        tf.keras.Model: The trained Conv-LSTM model.
        DataSplits: A named tuple containing the training, validation, and test datasets along with
                    their means and standard deviations.
    """

    # # Split the provided data into training, validation, and test sets
    # data_splits = train_val_test_split(data[features])
    #
    # # Extract the individual datasets and normalization parameters
    # train_data, val_data, test_data, train_mean, train_std = (
    #     data_splits.train_data,
    #     data_splits.val_data,
    #     data_splits.test_data,
    #     data_splits.train_mean,
    #     data_splits.train_std,
    # )

    # Generate a windowed representation of the data suitable for time series forecasting
    window = generate_window(train_data, val_data, test_data, days_in_past)

    # Determine the number of features in the windowed training dataset
    num_features = window.train_df.shape[1]

    # Create the Conv-LSTM model tailored to the data's feature count and the defined historical window size
    model = create_model(num_features, days_in_past)

    # Compile and train the model using the windowed data
    history = compile_and_fit(model, window)

    return window, model

