# Extracting Historical Data of Trading Pairs From Binance
# - J.D

import requests
import json
import time
from datetime import datetime
import sys
import os
import concurrent.futures
from queue import Queue
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse

# Returns list of symbols available on Binance which ends with a specific value
def get_symbols(endswith):
    """
    Gets all symbols from Binance which end with a specific string.

    Args:
        endswith: Ending string to query.
    
    Returns:
        A list of all symbols in the Binance historical database that end with the target string.
        Note: this includes symbols that are no longer tradeable in Binance.
    """
    base_url = 'https://api.binance.com'
    endpoint = '/api/v3/exchangeInfo'
    response = requests.get(base_url + endpoint)
    if response.status_code == 200:
        data = response.json()
        symbols = [symbol['symbol'] for symbol in data['symbols'] if symbol['symbol'].endswith(endswith)]
        return symbols
    else:
        sys.stderr.write(f"Error (SYMBOLS): {response.status_code}\n")

# Requests maximum allowed historical data from Binance using some symbol and granularity over a specified start and end time 
def get_historical_data(symbol, interval, start_time, end_time):
    """
    Retrieves historical data from Binance using their API.

    Args:
        symbol: The target symbol.
        interval: The interval of the kline data.
        start_time: The start of the historical data to retrieve.
        end_time: The end of the historical data to retrieve.

    Returns:
        The data (JSON) returned by Binance.
        Note: Binance has a limit of 1000 datapoints within a single request. 
    """
    base_url = 'https://api.binance.com'
    endpoint = '/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000
    }
    response = requests.get(base_url + endpoint, params=params)
    if response.status_code == 200:
        data = json.loads(response.text)
        return data
    else:
        sys.stderr.write(f"Error (HISTORICAL): {response.status_code}\n")

# Saves data returned by Binance to parquet
def save_to_parquet(data, filename):
    """
    Saves the historical data returned by Binance to a parquet file.

    Args:
        data: The historical data in Binance format.
        filename: The parquet file to save to.
    """

    columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    dtypes = ['datetime64[ms]', 'float64', 'float64', 'float64', 'float64', 'float64', 'datetime64[ms]', 'float64', 'int64', 'float64', 'float64', 'int64']
    
    df = pd.DataFrame(data, columns=columns)
    for i, dtype in enumerate(dtypes):
        df[columns[i]] = df[columns[i]].astype(dtype) 

    table = pa.Table.from_pandas(df)
    if os.path.exists(filename):
        existing_table = pq.read_table(filename)
        table = pa.concat_tables([existing_table, table])
    pq.write_table(table, filename)

def fetch_and_save_data(symbol, interval, folder="./"):
    """
    Wrapper function to fetch and save all historical data for a symbol within Binance.

    Args:
        symbol: The target symbol.
        interval: The interval of the kline data.
        folder: Target folder to save the historical data.

    Note: The historical data will be saved as {symbol}.parquet within the target folder.
    """
    end_t = int(time.time() * 1000)
    try:
        data = get_historical_data(symbol, interval, 0, end_t)
        filename = os.path.join(folder, f"{symbol}.parquet")
        if not os.path.exists(folder):
            os.makedirs(folder)
        while os.path.exists(filename):
            base_name, extension = os.path.splitext(filename)
            filename = f"{base_name} (1){extension}"

        while data:
            save_to_parquet(data, filename)
            next_start_time = int(data[-1][6]) + 1
            print(symbol, datetime.fromtimestamp(next_start_time/1000).strftime('%Y-%m-%d %H:%M:%S'))
            data = get_historical_data(symbol, interval, next_start_time, end_t)

        print(f"Got historical data for {symbol} at {interval}")
    except Exception as e:
        sys.stderr.write(f"Error fetching data for {symbol} at {interval}: {e}\n")

def process_symbol_queue(symbol_queue, args):
    """
    Calls the fetch_and_save_data() function for a symbol in the queue.

    Args:
        symbol_queue: Queue of symbols (to retrieve historical data)
        args: Input args retrieved by argparse.
    """
    while not symbol_queue.empty():
        symbol = symbol_queue.get()
        fetch_and_save_data(symbol, args.interval, args.folder)
        symbol_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description='Process historical data of trading pairs from Binance.')
    parser.add_argument('--interval', type=str, default='1h', help='Interval for historical data (e.g., 1m, 5m, 1h).')
    parser.add_argument('--folder', type=str, default='crypto-parquet-1h', help='Folder name to save the files.')
    args = parser.parse_args()

    all_symbols = get_symbols("USDT")  # Gets all crypto trading against USDT

    # ==================================== #
    # Multiprocessing for faster retrieval #

    symbol_queue = Queue()

    for symbol in all_symbols:
        symbol_queue.put(symbol)

    # Adjust to how many to retrieve in parallel
    num_workers = 6

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        worker_tasks = [executor.submit(process_symbol_queue, symbol_queue, args) for _ in range(num_workers)]

        for future in concurrent.futures.as_completed(worker_tasks):
            future.result()

    symbol_queue.join()

if __name__ == "__main__":
    main()