import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import time
from tqdm import tqdm
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

def read_parquet(file, threshold:float=0):
    """
    Reads a parquet file for a cryptocurrency dataset from Binance.

    Args:
        file: Parquet file containing Binance historical data.

    Returns:
        The cryptocurrency name and the historical data as a dataframe.
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

    return crypto_name, dataset

def train_model_incrementally(file_paths, folder_output, model_name, test_name=None, n_est=100, threshold:float=0, save_increments=True):
    """
    Incrementally trains a random forest model that predicts the up or down movement of a cryptocurrency using crypto historical data.

    Args:
        filepaths: A list of filepaths to the historical data. The model will be incrementally trained on these historical datasets.
        folder_output: Target folder for the saved random forest model.
        model_name: Name of saved random forest model (should be .pkl)
        test_name: Filename to save simple test results. Leave as None to not save tests.
        n_est: Number of estimators for the random forest model.
        save_increments: Whether to save each increment when traving the model.

    The created model requires the features 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 
    'SMA_5' (simple moving average at 5 timesteps), 'EMA_10' (exponential moving average at 10 timesteps),
    'RSI_14' (relative strength index at 14 timesteps), 'ATR_14' (average true range at 14 timesteps), and
    'CCI_20' (commodity channel index at 20 timesteps)
    """
    
    os.makedirs(folder_output, exist_ok=True)

    model = RandomForestClassifier(n_estimators=n_est, random_state=17, warm_start=True)
    n_est_total = 0
    count = 0
    
    with tqdm(total=len(file_paths)) as pbar:
        for crypto in file_paths:
            count += 1
            n_est_total += n_est // len(file_paths)
            name, dataset = read_parquet(crypto, threshold=threshold)
            print(f"Training on: {name}, Adding {n_est // len(file_paths)} Estimators, Total Estimators: {n_est_total}")

            # features = ['open_price', 'high_price', 'low_price','close_price', 'volume', 'SMA_5', 'EMA_10', 'RSI_14', 'ATR_14', 'CCI_20']
            features = ['close_price', 'volume', 'SMA_5', 'EMA_10', 'RSI_14', 'ATR_14', 'CCI_20']
            target = "movement"

            # Split data into features (X) and target variable (y)
            X = dataset[features][:-1]
            y = dataset[target][1:]

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
            
            model.set_params(n_estimators=n_est_total)
            model.fit(X_train, y_train)

            if len(file_paths) > 1 and save_increments:
                # Save model after iteration
                joblib.dump(model, os.path.join(folder_output, f"(after_{count}-{len(file_paths)}_{name})_{model_name}"))

            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            print("Accuracy:", accuracy)

            total = len(y_test)
            total_0 = 0

            for m in y_test:
                if m == 0:
                    total_0 += 1
            
            total_1 = total - total_0

            correct_0 = 0
            correct_1 = 0
            for a, b in zip(y_test, predictions):
                if a == 0 and b == 0:
                    correct_0 += 1
                if a == 1 and b == 1:
                    correct_1 += 1

            if test_name != None:
                # Save evaluation metrics to file
                with open(os.path.join(folder_output, test_name), "a+") as f:  # Open in append mode
                    f.seek(0)  # Move to the beginning of the file
                    content = f.read()  # Check if content exists
                    if not content:  # If the file is empty
                        f.write("Test Accuracy / Correct Down / Correct Up\n")  # Write header if empty
                        f.write(f"Datasets: {file_paths}\n")

                    f.write(f"After {name} ({count}/{len(file_paths)})\n")
                    f.write(f"{accuracy}, {correct_0 / total_0 * 100}%, {correct_1 / total_1 * 100}%\n")
            
            pbar.update(1)
    
    if model is not None:
        # Save the final model
        joblib.dump(model, os.path.join(folder_output, model_name))

def main():
    parser = argparse.ArgumentParser(description='Train a random forest model to predict cryptocurrency up and down movements.')
    parser.add_argument('--dataset', required=True, type=str, default=None, help='Historical cryptocurrency dataset in parquet format or folder of datasets.')
    parser.add_argument('--output_folder', type=str, default="./", help='Output folder to save random forest data.')
    parser.add_argument('--model_name', type=str, default="model.pkl", help='Filename to save random forest model (should be .pkl).')
    parser.add_argument('--test_name', type=str, default=None, help='Filename to save simple test results.')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of estimators for the random forest model.')
    parser.add_argument('--threshold', type=float, default=0, help='Percentage value to set threshold of movement (returns 1 for above and 0 for below).')
    parser.add_argument('--no_save_increments', action='store_false', help='Don\'t save incremental models.')
    args = parser.parse_args()
    
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

    if not args.model_name.endswith(".pkl"):
        print("Model name should end with .pkl")
        sys.exit()
    
    # ======= Create Models ======= #

    if willTrain:
        print("Training random forest model.")
        train_model_incrementally(file_paths, args.output_folder, args.model_name, args.test_name, n_est=args.n_estimators, threshold=args.threshold, save_increments=args.no_save_increments)
    else:
        print("Dataset path is not valid.")

if __name__ == "__main__":
    main()