import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from math import ceil
import argparse

def normalize_dataset(dataset):
    """
    Normalizes dataset using Min-Max Scaling except for the 'timestamp' column.

    Args:
        dataset: Pandas DataFrame containing the dataset.

    Returns:
        Normalized dataset.
    """
    timestamp_column = dataset['timestamp']  # Extract timestamp column
    non_timestamp_data = dataset.drop(columns=['timestamp'])  # Drop timestamp column
    min_vals = non_timestamp_data.min()
    max_vals = non_timestamp_data.max()
    normalized_data = (non_timestamp_data - min_vals) / (max_vals - min_vals)
    normalized_dataset = pd.concat([timestamp_column, normalized_data], axis=1)  # Concatenate timestamp column back
    return normalized_dataset

# Relative Strength Index (RSI)
def calculate_rsi(close_prices, window=14):
    """
    Calculates Relative Strength Index for kline data.

    Args:
        close_prices: Closing prices of historical data.
        window: Period to consider for RSI.

    Returns:
        RSI values.
    """
    diff = close_prices.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    ema_up = up.ewm(alpha=1/window, min_periods=window).mean()
    ema_down = down.ewm(alpha=1/window, min_periods=window).mean()
    rsi = 100 * ema_up / (ema_up + ema_down)
    return rsi

# Average True Range (ATR)
def calculate_atr(high, low, close, window=14):
    """
    Calculates Average True Range for kline data.

    Args:
        high: High prices of historical data.
        low: Low prices of historical data.
        close: Closing prices of historical data.
        window: Period to consider for ATR.

    Returns:
        ATR values.
    """
    tr = pd.concat([high - low,
                   (high - close.shift(1)).abs(),
                   (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/window, min_periods=window).mean()
    return atr

# Commodity Channel Index (CCI)
def calculate_cci(high, low, close, window=20):
    """
    Calculates Commodity Channel Index for kline data.

    Args:
        high: High prices of historical data.
        low: Low prices of historical data.
        close: Closing prices of historical data.
        window: Period to consider for CCI.

    Returns:
        CCI values.
    """
    typical_price = (high + low + close) / 3
    moving_average = typical_price.rolling(window=window).mean()
    mean_deviation = abs(typical_price - moving_average).rolling(window=window).mean()
    cci = (typical_price - moving_average) / (0.015 * mean_deviation)
    return cci

def read_parquet(file, sequence_length=1, threshold:float=0, max_rows=50000000, split=False):
    """
    Reads a parquet file for a cryptocurrency dataset from Binance and splits it into multiple parts.

    Args:
        file: Parquet file containing Binance historical data.
        sequence_length: Sequence length for LSTM model. Used to calculate split.
        max_rows: The max rows for each sub-part. Used to calculate split.
        split: Whether to split.

    Returns cryptocurrency name and list of subparts of the historical data.
    Note: The list of parts will be a multiple of 10. Even if split is false, the returned list will have a length of at least 10.
    """
    if not os.path.exists(file):
        print(f"The file '{file}' does not exist.")
        return

    crypto_name = os.path.splitext(os.path.basename(file))[0]

    dataset = pd.read_parquet(file)

    # Rename columns
    dataset.columns = ['timestamp', 'open_price', 'high_price', 'low_price', 'close_price',
                        'volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'unused']
    
    columns_to_save = ['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']

    dataset = dataset[columns_to_save] # save only minimum useful information
    
    # ========= Create Indicators ========= #

    # Simple Moving Average (SMA)
    dataset['SMA_5'] = dataset['close_price'].rolling(window=5).mean()
    # Exponential Moving Average (EMA)
    dataset['EMA_10'] = dataset['close_price'].ewm(alpha=0.1, min_periods=10).mean()
    # Relative Strength Index (RSI)
    dataset['RSI_14'] = calculate_rsi(dataset['close_price'])
    # Average True Range (ATR)
    dataset['ATR_14'] = calculate_atr(dataset['high_price'], dataset['low_price'], dataset['close_price'])
    # Commodity Channel Index (CCI)
    dataset['CCI_20'] = calculate_cci(dataset['high_price'], dataset['low_price'], dataset['close_price'])
    
    # ========= Indicators END ========= #

    # Fill Na
    dataset.ffill(inplace=True)
    dataset.dropna(inplace=True)

    # Normalize dataset
    dataset = normalize_dataset(dataset)

    percent = threshold / 100

    # Create Movement Indicator
    dataset['movement'] = np.where((dataset['close_price'] - dataset['open_price']) / dataset['open_price'] >= percent, 1, 0)
    dataset['movement'] = [ [0, 1] if val == 1 else [1, 0] for val in dataset['movement'] ]

    if split:
        total_rows = sequence_length * dataset.shape[0]

        # Calculate the original number of parts
        original_num_parts = ceil(total_rows / max_rows)

        # Make sure num_parts is a multiple of ten but never below the original value
        adjusted_num_parts = ceil(original_num_parts / 10) * 10
    else:
        adjusted_num_parts = 10

    # Split dataset into parts
    split_parts = np.array_split(dataset, adjusted_num_parts)
    returned_parts = []
    for part in split_parts:
        returned_parts.append(part.to_dict(orient='list'))

    return crypto_name, returned_parts

def split_list_percentages(original_list):
    """
    Splits a list into 2 new lists with 80% and 20% of the items.

    Args:
        original_list: The list to be split.

    Returns:
        A tuple containing two lists: 80% list, and 20% list.
    """
    n = len(original_list)
    split_point1 = int(n * 0.8)  # Index after 80% of the list

    list1 = original_list[:split_point1]  # First 80%
    list2 = original_list[split_point1:]  # Last 20%

    return list1, list2

def split_dict_by_percentage(data_dict, percentage):
    """Splits a dictionary into two dictionaries with a specific percentage split,
    preserving time-sequence order within features.

    Args:
        data_dict: The original dictionary to split.
        percentage: The percentage of data to include in the first split (0.0 to 1.0).

    Returns:
        A tuple containing two dictionaries: the first with 'percentage' data and
        the second with the remaining data, both maintaining time-sequence order.
    """
    if not 0 <= percentage <= 1:
        raise ValueError("Percentage must be between 0 and 1")

    split_index = int(len(data_dict[list(data_dict.keys())[0]]) * percentage)
    dict1 = {key: value[:split_index] for key, value in data_dict.items()}
    dict2 = {key: value[split_index:] for key, value in data_dict.items()}
    return dict1, dict2

def merge_dicts_optimized(dict_list):
    """
    Merges a list of dictionaries with the same keys (containing lists) into a single dictionary.

    Args:
        dict_list: A list of dictionaries with the same keys.

    Returns:
        A single dictionary with merged values for each key.
    """
    merged_dict = {key: [] for key in dict_list[0]}
    for key in dict_list[0]:
        for subdict in dict_list:
            merged_dict[key].extend(subdict[key])
    return merged_dict

def build_merged_dict(dict_list):
    """
    Builds a dictionary by calling merge_dicts_optimized.

    Args:
        dict_list: A list of dictionaries with the same keys.

    Returns:
        A single dictionary with merged values for each key.
    """
    return merge_dicts_optimized(dict_list)

def reshape_data(stock_data, sequence_length=1):
    """
    Reshapes data for a single stock into a format suitable for LSTM training.

    Args:
        stock_data: Dictionary containing data for a single stock.
        sequence_length: Number of timesteps to include in each training sample.

    Returns:
        A tuple containing three elements:
            - reshaped_data: An array of shape (num_samples, sequence_length,
    """
    # Get features and timestamps

    features = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'SMA_5', 'EMA_10', 'RSI_14', 'ATR_14', 'CCI_20']
    timestamps = stock_data['timestamp']

    # Initialize lists to store data
    data_list = []
    timestamp_list = []

    # Iterate over data entries with a sliding window
    for i in range(len(stock_data[features[0]]) - sequence_length + 1):
        # Get a sequence of data points for each feature
        sequence = []
        for feature_name in features:
            sequence.append(stock_data[feature_name][i:i+sequence_length])
        # Stack the features into a single array
        stacked_sequence = np.stack(sequence, axis=1)

        # Add the sequence and timestamp to respective lists
        data_list.append(stacked_sequence)
        timestamp_list.append(timestamps[i])

    # Convert lists to NumPy arrays
    reshaped_data = np.array(data_list)
    timestamps = np.array(timestamp_list)

    return reshaped_data, timestamps, None  # You can return original data here if needed

def build_LSTM_model(num_features, sequence_length):
    """
    Creates an appropriate simple LSTM model

    Args:
        num_features: the number of features in the dataset
        sequence_length: Number of timesteps in each training sample.

    Returns:
        An LSTM model
    """
    # Define the LSTM model
    model = Sequential([
        Input(shape=(sequence_length, num_features)),
        LSTM(units=50, return_sequences=False),
        Dropout(rate=0.2),
        Dense(units=2, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"]
    )
    return model

def train_model_incrementally(file_paths, folder_output, model_name, test_name=None, sequence_length=3, epochs=10, batch_size=32, threshold:float=0, split=False, save_increments=True):
    """
    Incrementally trains an LSTM model that predicts the closing price of a cryptocurrency using crypto historical data.

    Args:
        filepaths: A list of filepaths to the historical data. The model will be incrementally trained on these historical datasets.
        folder_output: Target folder for the saved LSTM model.
        model_name: Name of saved LSTM model (should be .keras)
        test_name: Filename to save simple test results. Leave as None to not save tests.
        sequence_length: The number of timestamps within a sequence.
        epochs: Number of epochs to train model per increment.
        batch_size: Size of batch during training.
        split: Whether to split individual historical data files to multiple sets. Set this to True when facing memory issues.
            Note: Splitting may affect the trained model.
        save_increments: Whether to save each increment when traving the model.

    The created model requires the features 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 
    'SMA_5' (simple moving average at 5 timesteps), 'EMA_10' (exponential moving average at 10 timesteps),
    'RSI_14' (relative strength index at 14 timesteps), 'ATR_14' (average true range at 14 timesteps), and
    'CCI_20' (commodity channel index at 20 timesteps)
    """
    num_features = 0
    model = None
    count = 0

    os.makedirs(folder_output, exist_ok=True)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f"{folder_output}/logs")

    for crypto in file_paths:
        count += 1
        name, datasets = read_parquet(crypto, sequence_length=sequence_length, threshold=threshold, split=split)
        print(f"Training on: {name}")

        train, tests = split_list_percentages(datasets)
        del datasets
        if split == False:
            train = [build_merged_dict(train)]
        test_merged = build_merged_dict(tests)
        del tests

        # Define EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        for part_num, part_data in enumerate(train, start=1):
            print(f"Training on part {part_num}/{len(train)} of {name}")

            train_set, validation_set = split_dict_by_percentage(part_data, 0.8)
            movement = train_set['movement']
            train_reshaped, _, _ = reshape_data(train_set, sequence_length=sequence_length)
            del train_set
            movement_validation = validation_set['movement']
            validation_reshaped, _, _ = reshape_data(validation_set, sequence_length=sequence_length)
            del validation_set

            # Update num_features if it's the first iteration
            if num_features == 0:
                num_features = train_reshaped.shape[2]
                model = build_LSTM_model(num_features, sequence_length=sequence_length)  # Rebuild with features

            # Prepare training data from past data
            X_train = train_reshaped[:-1, :, :]  # All Time Sequence Groups Except Last One
            y_train = np.array(movement[sequence_length:])  # Align Movement (Next Rank For Previous Time Sequence Group)
            
            del train_reshaped

            # Prepare validation data
            X_val = validation_reshaped[:-1, :, :]
            y_val = np.array(movement_validation[sequence_length:])

            del validation_reshaped

            # Train the model on the prepared data
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping, tensorboard_callback])
            
            del X_train
            del y_train
            del X_val
            del y_val

        if len(file_paths) > 1 and save_increments:
            # Save model after iteration
            model.save(os.path.join(folder_output, f"(after_{count}-{len(file_paths)}_{name})_{model_name}"))

        # Evaluate the model on the test set
        movement_test = test_merged['movement']
        test, _, _ = reshape_data(test_merged, sequence_length=sequence_length)
        del test_merged
        X_test = test[:-1, :, :]
        y_test = np.array(movement_test[sequence_length:])
        del test
        test_metrics = model.evaluate(X_test, y_test, callbacks=[tensorboard_callback])
        predictions = model.predict(X_test)

        del X_test
        
        total = len(y_test)
        total_0 = 0

        for m in y_test:
            if m[0] == 1:
                total_0 += 1
        
        total_1 = total - total_0

        correct_0 = 0
        correct_1 = 0
        for a, b in zip(y_test, predictions):
            if a[0] == 1 and b[0] > 0.5:
                correct_0 += 1
            if a[1] == 1 and b[1] > 0.5:
                correct_1 += 1

        del y_test

        # Print the evaluation metrics
        print(f"Datasets: {file_paths}, After {name} ({count}/{len(file_paths)})\n")
        print(f"{test_metrics}\n")

        if test_name != None:
            # Save evaluation metrics to file
            with open(os.path.join(folder_output, test_name), "a+") as f:  # Open in append mode
                f.seek(0)  # Move to the beginning of the file
                content = f.read()  # Check if content exists
                if not content:  # If the file is empty
                    f.write("Test Loss / Accuracy / Correct Total / Correct Down / Correct Up\n")  # Write header if empty
                    f.write(f"Datasets: {file_paths}\n")

                f.write(f"After {name} ({count}/{len(file_paths)})\n")
                f.write(f"{test_metrics}, {(correct_0 + correct_1) / total * 100}%, {correct_0 / total_0 * 100}%, {correct_1 / total_1 * 100}%\n")

    if model is not None:
        # Save the final model
        model.save(os.path.join(folder_output, model_name))

def main():
    parser = argparse.ArgumentParser(description='Train an LSTM model to predict cryptocurrency closing data.')
    parser.add_argument('--dataset', required=True, type=str, default=None, help='Historical cryptocurrency dataset in parquet format or folder of datasets.')
    parser.add_argument('--output_folder', type=str, default="./", help='Output folder to save LSTM data.')
    parser.add_argument('--model_name', type=str, default="model.keras", help='Filename to save LSTM model (should be .keras).')
    parser.add_argument('--test_name', type=str, default=None, help='Filename to save simple test results.')
    parser.add_argument('--sequence', type=int, default=3, help='Sequence length to train LSTM model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train model per increment.')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of batch during training.')
    parser.add_argument('--threshold', type=float, default=0, help='Percentage value to set threshold of movement (returns 1 for above and 0 for below).')
    parser.add_argument('--split', action='store_true', help='Split individual historical data files to multiple sets. Use this argument when facing memory issues.')
    parser.add_argument('--no_save_increments', action='store_false', help='Don\'t save incremental models.')
    args = parser.parse_args()

    tf.random.set_seed(17)

    print("TensorFlow version:", tf.__version__)

    willTrain = False

    if args.dataset is None:
        willTrain = False
    elif os.path.isfile(args.dataset):
        willTrain = True
        file_paths = []
        file_paths.append(args.dataset)
    elif os.path.exists(args.dataset):
        willTrain = True
        file_paths = []
        directory = args.dataset
        # List all items in the directory
        items = os.listdir(directory)
        # Iterate over each item
        for item in sorted(items):
            if not os.path.isfile(os.path.join(args.dataset, item)):
                continue
            if not item.endswith(".parquet"):
                continue
            # Join the directory path with the item to get the full file path
            item_path = os.path.join(directory, item)
            # Append the file path to the list
            file_paths.append(item_path)
    else:
        willTrain = False

    if not args.model_name.endswith(".keras"):
        print("Model name should end with .keras")
        sys.exit()
    
    # ======= Create Models ======= #

    if willTrain:
        print("Training LSTM model.")
        train_model_incrementally(file_paths, args.output_folder, args.model_name, args.test_name, sequence_length=args.sequence, epochs=args.epochs, batch_size=args.batch_size, threshold=args.threshold, split=args.split, save_increments=args.no_save_increments)
    else:
        print("Dataset path is not valid.")

if __name__ == "__main__":
    main()