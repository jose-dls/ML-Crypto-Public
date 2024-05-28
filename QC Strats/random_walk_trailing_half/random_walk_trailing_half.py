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
#            Random Walk With Half Sell Trailing Stop Loss Strategy            #
#                                                                              #
# This strategy serves as a benchmark strategy and buys a portfolio every hour #
# indiscriminately regardless of the conditions of the asset. When buying, all #
# available funds are used to open a position and are split equally among the  #
# cryptocurrencies to buy. A trailing stop loss is applied for half sell       #
# orders (half of the available  assets of a cryptocurrency are sold once a    #
# 1.5% trailing loss is hit).                                                  #
################################################################################

class CryptoLSTMStrategy(QCAlgorithm):
    def Initialize(self):
        self.symbols_a = ["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LTCUSDT", "SHIBUSDT", "SOLUSDT", "XRPUSDT"]  # Crypto symbols to trade (replace with your desired symbols)

        self.SetStartDate(2024, 1, 1)  # Set start date for backtest
        self.SetEndDate(2024, 1, 31)  # Set end date for backtest
        
        self.starting_equity = 10000
        self.set_account_currency("USDT", self.starting_equity)
        self.starting_equity = self.portfolio.cash_book["USDT"].amount  # Just in case reported value is lower than starting equity
        self.sequence_length = 1  # Number of data points for prediction
        self.res = Resolution.Hour # Resolution - Make sure to change schedule as well
        self.symbols = []
        self.settings.FreePortfolioValue = self.portfolio.cash_book["USDT"].amount * 0.05
        self.set_brokerage_model(BrokerageName.BINANCE, AccountType.CASH)
        temp_cash = self.portfolio.cash_book["USDT"].amount
        self.Debug(f"Starting USDT: {temp_cash}")
        self.count = 0
        self.allocation = 5
        self.holdings = set()
        self.local_max_price = {}
        self.stop_price = {}
        self.sell_stop = 0.985  # Sell after 1.5% Loss
        for symbol in self.symbols_a:
            try:
                self.AddCrypto(symbol, resolution=self.res)
                self.symbols.append(symbol)
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
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(TimeSpan.FromMinutes(5)), self.CheckLoss)
    
    def Trade(self):
        self.count += 1
        
        # Buy Crypto
        for symbol in self.symbols:
            self.buy_crypto_holdings(symbol, 1/len(self.symbols))

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
        if self.getting_price:
            return self.starting_equity
        balance = 0
        count = 0
        for s in symbols:
            if self.starting_prices[s] > 0:
                count += 1
        
        for s in symbols:
            if self.starting_prices[s] > 0:
                symbol = self.Securities[s]
                balance += (self.starting_equity / count) * (symbol.Price / self.starting_prices[s])
        
        return balance

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