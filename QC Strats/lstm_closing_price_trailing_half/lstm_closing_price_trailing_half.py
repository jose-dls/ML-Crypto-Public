# region imports
from AlgorithmImports import *
import tensorflow as tf
import io
import math
from math import floor
from datetime import date, timedelta
import pandas as pd
import numpy as np
from scipy import stats
# endregion

################################################################################
#   LSTM Closing Price Prediction With Half Sell Trailing Stop Loss Strategy   #
#                                                                              #
# This strategy predicts hourly and buys cryptocurrencies predicted to         #
# increase above a certain percentage. The error of each prediction (per       #
# cryptocurrency) is saved (max 100) for better accuracy. Cryptocurrencies     #
# are bought at a fix allocation (1/5 of starting equity). A trailing stop     #
# loss is applied for half sell orders (half of the available assets of a      #
# cryptocurrency are sold once a 1.5% trailing loss is hit).                   #
################################################################################

class CryptoLSTMStrategy(QCAlgorithm):
    def Initialize(self):
        self.model = self.LoadModel("/model.keras") # Load model
        self.symbols_a = ["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LTCUSDT", "SHIBUSDT", "SOLUSDT", "XRPUSDT"]  # Crypto symbols to trade (replace with your desired symbols)

        self.SetStartDate(2024, 1, 1)  # Set start date for backtest
        self.SetEndDate(2024, 1, 31)  # Set end date for backtest
        self.set_warm_up(timedelta(hours=100)) # Set a warmup period to populate prediction errors
        
        self.starting_equity = 10000
        self.set_account_currency("USDT", self.starting_equity)
        self.starting_equity = self.portfolio.cash_book["USDT"].amount  # Just in case reported value is lower than starting equity
        self.sequence_length = 24  # Number of data points for prediction
        self.res = Resolution.Hour # Resolution - Make sure to change schedule as well
        self.symbols = []
        self.settings.FreePortfolioValue = self.portfolio.cash_book["USDT"].amount * 0.05
        self.set_brokerage_model(BrokerageName.BINANCE, AccountType.CASH)
        temp_cash = self.portfolio.cash_book["USDT"].amount
        self.Debug(f"Starting USDT: {temp_cash}")
        self.count = 0
        self.allocation = 5
        # Store last prediction for each symbol
        self.last_prediction = {}
        self.prediction_errors = {}
        self.holdings = set()
        self.historical_min_data = {}
        self.historical_max_data = {}
        self.historical_min_price = {}
        self.historical_max_price = {}
        self.local_max_price = {}
        self.stop_price = {}
        self.sell_stop = 0.985  # Sell after 1.5% Loss
        for symbol in self.symbols_a:
            try:
                self.AddCrypto(symbol, resolution=self.res)
                self.symbols.append(symbol)
                self.last_prediction[symbol] = None
                self.prediction_errors[symbol] = []
                self.historical_min_data[symbol] = None
                self.historical_max_data[symbol] = None
                self.historical_min_price[symbol] = None
                self.historical_max_price[symbol] = None
                self.local_max_price[symbol] = None
                self.stop_price[symbol] = None
            except:
                self.Debug(f"Unable to add: {symbol}")
        self.getting_price = True
        self.starting_prices = {}
        # Set Benchmark
        self.SetBenchmark(lambda x: self.CustomBenchmark(self.symbols))
        # Handle the error appropriately, e.g., stop the algorithm
        # Schedule function to run every hour
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(TimeSpan.FromHours(1)), self.Trade)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(TimeSpan.FromMinutes(1)), self.CheckLoss)
    
    def Trade(self):
        # List to store symbols to buy 
        symbols_to_buy = []

        self.Debug(f"Making Predictions. Count {self.count}")
        for s in self.symbols:
            symbol = self.Securities[s]
            try:
                # Get rolling window data (adjust resolution as needed)
                symbol_data = self.History(symbol.Symbol, self.sequence_length + 50, self.res)  # Extra 50 for calculations
                # Ensure enough data for analysis (optional)
                if symbol_data.shape[0] != self.sequence_length + 50:
                    continue
                desired_columns = ["open", "high", "low", "close", "volume"]
                data_array = symbol_data[desired_columns]

                # ========= Create Indicators ========= #
                # Simple Moving Average (SMA)
                data_array["SMA_5"] = data_array["close"].rolling(window=5).mean()
                # Exponential Moving Average (EMA)
                data_array["EMA_10"] = data_array["close"].ewm(alpha=0.1, min_periods=10).mean()
                # Relative Strength Index (RSI)
                data_array["RSI_14"] = self.calculate_rsi(data_array["close"])
                # Average True Range (ATR)
                data_array["ATR_14"] = self.calculate_atr(data_array["high"], data_array["low"], data_array["close"])
                # Commodity Channel Index (CCI)
                data_array["CCI_20"] = self.calculate_cci(data_array["high"], data_array["low"], data_array["close"])
                
                # ========= Indicators END ========= #

                data_array = data_array[-self.sequence_length:]

                # Normalize data using QuantConnect functions (replace with your normalization method if needed)
                normalized_data = self.NormalizeData(data_array, s)

                # Reshape data for LSTM model (replace with your logic if needed)
                reshaped_data = self.ReshapeData(normalized_data)

                # Make prediction using the model
                predicted_close = self.model.predict(reshaped_data)[0][0]  # Assuming single output

                # Skip if no prediction
                if predicted_close is None or math.isnan(predicted_close):
                    continue
                
                # Update Prediction Errors
                if self.last_prediction[s] is not None:
                    self.prediction_errors[s].append(self.last_prediction[s] - self.NormalizedPrice(s))
                    self.prediction_errors[s] = self.prediction_errors[s][-100:]

                # Update Last Prediciton
                self.last_prediction[s] = predicted_close

                # Calculate Buy Indicators
                if len(self.prediction_errors[s]) >= 3:
                    filtered_data = self.filter_outliers_zscore(self.prediction_errors[s])
                    # filtered_data = self.prediction_errors[s]
                    error = sum(filtered_data) / len(filtered_data)
                    if predicted_close - error > self.NormalizedPrice(s) * 1.005: # Buy When More Than 0.5% Gain
                        symbols_to_buy.append(s)
            except Exception as e:
                self.Error(e)
                continue
        
        self.count += 1

        # Buy Crypto
        for symbol in symbols_to_buy:
            self.buy_crypto_holdings(symbol, (self.starting_equity / self.allocation) / self.portfolio.cash_book["USDT"].amount)  # Adjust for price

        self.Debug(f"Finished. Count {self.count}")
       
    def CheckLoss(self):
        for s in self.symbols:
            if s in self.holdings:
                symbol = self.Securities[s]
                minute_data = self.History(symbol.Symbol, 1, Resolution.Minute)
                if minute_data.shape[0] < 1:
                    continue
                minute_high = max(minute_data['high'])
                if self.local_max_price[s] is None:
                    self.local_max_price[s] = minute_high
                    self.stop_price[s] = self.local_max_price[s] * self.sell_stop
                elif minute_high > self.local_max_price[s]:
                    self.local_max_price[s] = minute_high
                    self.stop_price[s] = self.local_max_price[s] * self.sell_stop
                
                if self.stop_price[s] is None:
                    if self.local_max_price[s] is None:
                        self.local_max_price[s] = minute_high
                        self.stop_price[s] = self.local_max_price[s] * self.sell_stop
                    else:
                        self.stop_price[s] = self.local_max_price[s] * self.sell_stop
                
                if minute_data['close'][-1] < self.stop_price[s]:  # Latest close is lower than stop price
                    # self.sell_crypto_holdings(s, conversion=minute_data['close'][-1])
                    # self.local_max_price[s] = None
                    # self.stop_price[s] = None  # Reset after sell
                    
                    self.sell_crypto_holdings(s, percentage=0.5, conversion=minute_data['close'][-1])
                    self.local_max_price[s] = minute_high
                    self.stop_price[s] = self.local_max_price[s] * self.sell_stop  # Reset after sell
    
    def OnData(self, data):
        if self.is_warming_up:
            return
        if self.getting_price:
            for s in self.symbols:
                symbol = self.Securities[s]
                self.starting_prices[s] = symbol.Price
                self.Debug(f"Starting Price of {s}: ")
                self.Debug(symbol.Price)
            self.getting_price = False
        return

    def CustomBenchmark(self, symbols):
        if self.getting_price or self.is_warming_up:
            return self.starting_equity
        balance = 0
        for s in symbols:
            symbol = self.Securities[s]
            balance += (self.starting_equity / len(symbols)) * (symbol.Price / self.starting_prices[s])
        return balance
    
    def on_warmup_finished(self):
        # Perform a prediction/buy after warmup
        self.Trade()

    def buy_crypto_holdings(self, symbol, percentage):
        """
            Symbol: Symbol to buy
            Percentage: Percentage of available balance to use
        """
        if self.is_warming_up:  # No buying during warmup
            return
        try:
            crypto = self.Securities[symbol]
            base_currency = crypto.BaseCurrency
            # Calculate the target quantity in the base currency
            # self.Debug(percentage)
            # self.Debug(self.portfolio.cash_book["USDT"].amount)
            # self.Debug(base_currency.ConversionRate)

            quantity = (percentage * (self.portfolio.cash_book["USDT"].amount - self.Settings.FreePortfolioValue)) / base_currency.ConversionRate

            # Round down to observe the lot size
            lot_size = crypto.SymbolProperties.LotSize
            quantity = quantity - (quantity % lot_size)

            if quantity <= 0:
                return
            balance = self.portfolio.cash_book["USDT"].amount
            self.Debug(f"Attempting to set {percentage} of balance to {symbol} by buying {quantity} of {symbol} (Current Balance {balance})")

            if (quantity * base_currency.ConversionRate) < (self.portfolio.cash_book["USDT"].amount - self.Settings.FreePortfolioValue) and self.is_valid_order_size(crypto, quantity):
                self.market_order(symbol, quantity)
                if symbol not in self.holdings:  # Update Local Max And Stop Price Only If Not Previously Holding
                    self.local_max_price[symbol] = base_currency.ConversionRate
                    self.stop_price[symbol] = self.local_max_price[symbol] * self.sell_stop
                elif base_currency.ConversionRate > self.local_max_price[symbol]:
                    self.local_max_price[symbol] = base_currency.ConversionRate
                    self.stop_price[symbol] = self.local_max_price[symbol] * self.sell_stop
                self.holdings.add(symbol)
                balance = self.portfolio.cash_book["USDT"].amount
                self.Debug(f"Bought {quantity} of {symbol} at USDT {quantity * base_currency.ConversionRate} (Total) (Current Balance: {balance})")
                self.Log(f"Bought {quantity} of {symbol} at USDT {quantity * base_currency.ConversionRate} (Total) (Current Balance: {balance})")
        except:
            self.Debug(f"Unable to perform trade on {symbol} (Buy)")

    # Brokerages have different order size rules
    # Binance considers the minimum volume (price x quantity):
    def is_valid_order_size(self, crypto, quantity):
        return abs(crypto.Price * quantity) > crypto.SymbolProperties.MinimumOrderSize

    def sell_crypto_holdings(self, symbol, percentage=1, conversion=None):
        """
            Symbol: Symbol to sell
            Percentage: Percentage of symbol holding to sell
        """
        if self.is_warming_up:  # No selling during warmup
            return
        try:
            crypto = self.securities[symbol]
            base_currency = crypto.base_currency

            if conversion is None:
                conversion_rate = base_currency.ConversionRate
            else:
                conversion_rate = conversion

            # Validate percentage (inclusive of 0 and 1)
            if percentage < 0 or percentage > 1:
                return

            # Calculate quantity to sell based on percentage
            quantity = crypto.holdings.quantity * percentage

            # Avoid negative amount after liquidate
            quantity = min(quantity, base_currency.amount)
                
            # Round down to observe the lot size
            lot_size = crypto.symbol_properties.lot_size
            quantity = floor(quantity / lot_size) * lot_size

            if quantity <= 0:
                return

            if self.is_valid_order_size(crypto, quantity):
                self.market_order(symbol, -quantity)
                if percentage == 1:
                    self.holdings.remove(symbol)
                    self.local_max_price[symbol] = None
                    self.stop_price[symbol] = None  # Reset after sell
                balance = self.portfolio.cash_book["USDT"].amount
                self.Debug(f"Sold {symbol} at USDT {quantity * conversion_rate} (Total) (Current Balance: {balance})")
                self.Log(f"Sold {symbol} at USDT {quantity * conversion_rate} (Total) (Current Balance: {balance})")
        except:
            self.Debug(f"Unable to perform trade on {symbol} (Sell)")
    
    def NormalizeData(self, data, symbol):
        """
        This function normalizes data using historical min/max values obtained from QuantConnect.
        """
        # Recalculate Historical Min and Max Every Month
        if self.historical_min_data[symbol] is None or self.historical_max_data[symbol] is None or self.count % 730 == 0:
            sec = self.Securities[symbol]
            # Get start date
            today = date.today()
            # Calculate the date 5 years ago (unable to directly get min/max values so must have a lookback period)
            five_years_ago = today - timedelta(days=365 * 5)
            # Get historical data for the entire period
            historical = self.History(sec.Symbol, five_years_ago, today, self.res)  # Adjust resolution as needed

            desired_columns = ["open", "high", "low", "close", "volume"]
            historical = historical[desired_columns]

            # ========= Create Indicators ========= #

            # Simple Moving Average (SMA)
            historical["SMA_5"] = historical["close"].rolling(window=5).mean()
            # Exponential Moving Average (EMA)
            historical["EMA_10"] = historical["close"].ewm(alpha=0.1, min_periods=10).mean()
            # Relative Strength Index (RSI)
            historical["RSI_14"] = self.calculate_rsi(historical["close"])
            # Average True Range (ATR)
            historical["ATR_14"] = self.calculate_atr(historical["high"], historical["low"], historical["close"])
            # Commodity Channel Index (CCI)
            historical["CCI_20"] = self.calculate_cci(historical["high"], historical["low"], historical["close"])
            
            # ========= Indicators END ========= #

            # Fill Na
            historical.ffill(inplace=True)
            historical.dropna(inplace=True)

            # Calculate historical min/max values (consider using rolling statistics for more flexibility)
            self.historical_min_data[symbol] = historical.min()
            self.historical_max_data[symbol] = historical.max()

        # Normalize data using historical min/max
        return (data - self.historical_min_data[symbol]) / (self.historical_max_data[symbol] - self.historical_min_data[symbol])

    def NormalizedPrice(self, symbol):
        sec = self.Securities[symbol]
        if self.historical_min_price[symbol] is None or self.historical_max_price[symbol] is None or self.count % 730 == 0:
            # Get start date
            today = date.today()
            # Calculate the date 5 years ago
            two_years_ago = today - timedelta(days=365 * 5)
            historical_data = self.History(sec.Symbol, two_years_ago, today, Resolution.Daily)  # Resolution Does Not Need To Be Precise For Max/Min Price
            self.historical_min_price[symbol] = historical_data["close"].values.min()
            self.historical_max_price[symbol] = historical_data["close"].values.max()
        return (sec.Price - self.historical_min_price[symbol]) / (self.historical_max_price[symbol] - self.historical_min_price[symbol])

    def ReshapeData(self, normalized_data):
        """
        This function reshapes data for the LSTM model.
        """
        # Assuming a 2D array for the model with shape (sequence_length, features)
        desired_columns = ["open", "high", "low", "close", "volume", "SMA_5", "EMA_10", "RSI_14", "ATR_14", "CCI_20"]
        return np.reshape(normalized_data[desired_columns].values, (1, self.sequence_length, len(desired_columns)))

    # Relative Strength Index (RSI)
    def calculate_rsi(self, close_prices, window=14):
        diff = close_prices.diff()
        up = diff.clip(lower=0)
        down = -diff.clip(upper=0)
        ema_up = up.ewm(alpha=1/window, min_periods=window).mean()
        ema_down = down.ewm(alpha=1/window, min_periods=window).mean()
        rsi = 100 * ema_up / (ema_up + ema_down)
        return rsi

    # Average True Range (ATR)
    def calculate_atr(self, high, low, close, window=14):
        tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/window, min_periods=window).mean()
        return atr

    # Commodity Channel Index (CCI)
    def calculate_cci(self, high, low, close, window=20):
        typical_price = (high + low + close) / 3
        moving_average = typical_price.rolling(window=window).mean()
        mean_deviation = abs(typical_price - moving_average).rolling(window=window).mean()
        cci = (typical_price - moving_average) / (0.015 * mean_deviation)
        return cci

    def filter_outliers_zscore(self, data, threshold=3):
        """
        Filters outliers from a list using Z-scores.

        Args:
            data (list): The list of data to filter.
            threshold (float, optional): The number of standard deviations 
                outside the mean to consider an outlier. Defaults to 3.

        Returns:
            list: A new list containing only non-outlier values.
        """
        mean = np.mean(data)
        std = np.std(data)
        z_scores = stats.zscore(data)
        filtered_data = [x for x, score in zip(data, z_scores) if abs(score) <= threshold]
        return filtered_data

    def LoadModel(self, file_name):
        # Load the machine learning model from the object store
        file_path = self.object_store.get_file_path(file_name)
        
        return tf.keras.models.load_model(file_path, compile=False)  # Return the loaded model