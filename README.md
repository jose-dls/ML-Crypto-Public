# ML-Crypto-Public
Exploration of machine learning techniques on cryptocurrency trading.

## Introduction

This README documents the process of using this repository to setup an LSTM model to predict future closing prices of cryptocurrencies. Sample algorithms for the QuantConnect platform have been provided in QC Strats.

## Downloading Historical Data

First download historical data from Binance using the historical.py script.

> python3 historical.py

or

> python historical.py

depending on your system configuration.

Note: this downloads all USDT pairs. Modify the all_symbols variable to a list of target symbols as required.

Example:

> all_symbols = ["BTCUSDT", "ETHUSDT"]

## Training Models

### Training The LSTM Model (Closing Price Prediction)

Once historical data has been downloaded, run the lstm_closing_price.py script to train an LSTM model.

Arguments:

--dataset: Historical cryptocurrency dataset in parquet format or folder of datasets.

--output_folder: Output folder to save LSTM data.

--model_name: Filename to save LSTM model (should be .keras).

--test_name: Filename to save simple test results.

--sequence: Sequence length to train LSTM model.

--epochs: Number of epochs to train model per increment.

--batch_size: Size of batch during training.

--split: Split individual historical data files to multiple sets. Use this argument when facing memory issues.

--no_save_increments: Don't save incremental models.

Example:

> python3 lstm_closing_price.py --dataset \<historical dataset> --output_folder \<target output folder>

or

> python lstm_closing_price.py --dataset \<historical dataset> --output_folder \<target output folder>

depending on your system configuration.

### Training The LSTM Model (Movement Prediction)

Training an LSTM model to predict price movement is done similarly by running the lstm_movement.py script. 

Arguments:

--dataset: Historical cryptocurrency dataset in parquet format or folder of datasets.

--output_folder: Output folder to save LSTM data.

--model_name: Filename to save LSTM model (should be .keras).

--test_name: Filename to save simple test results.

--sequence: Sequence length to train LSTM model.

--epochs: Number of epochs to train model per increment.

--batch_size: Size of batch during training.

--threshold: Percentage value to set threshold of movement (returns 1 for above and 0 for below).

--split: Split individual historical data files to multiple sets. Use this argument when facing memory issues.

--no_save_increments: Don't save incremental models.

Example:

> python3 lstm_movement.py --dataset \<historical dataset> --output_folder \<target output folder>

or

> python lstm_movement.py --dataset \<historical dataset> --output_folder \<target output folder>

depending on your system configuration.

### Training The Gradient Boosting and Random Forest Model (Movement Prediction)

You can also train a Gradient Boosting Classifier model or a Random Forest Classifier model to predict price movement. Just use the gradient_boosting_movement.py or random_forest_movement.py script. Please note that unlike the other models, the Random Forest models can have a large size depending on your training dataset.

Arguments:

--dataset: Historical cryptocurrency dataset in parquet format or folder of datasets.

--output_folder: Output folder to save LSTM data.

--model_name: Filename to save LSTM model (should be .keras).

--test_name: Filename to save simple test results.

--n_estimators: Number of estimators for the gradient boosting/random forest model.

--threshold: Percentage value to set threshold of movement (returns 1 for above and 0 for below).

--no_save_increments: Don't save incremental models.

Example:

> python3 gradient_boosting_movement.py --dataset \<historical dataset> --output_folder \<target output folder>

or

> python gradient_boosting_movement.py --dataset \<historical dataset> --output_folder \<target output folder>

depending on your system configuration.

### Note

Please note that as of this release, the default parameters for the movement models in general only have a slight improvement to random guessing. Please use them as a framework to suit your needs.

## Backtesting The Model

Sample algorithms have been provided in QC Strats. These algorithsm work in the QuantConnect platform, however, you can adapt them to your requirements. A brief overview to using these strategies in QuantConnect:

- Create an account.

- Upload your model/s to the Object Store.

- Create a project.

- Replace the contents of main.py with the contents of your desired strategy.

- Within the Initialize function, change the filepath parameter for the self.LoadModel() function assigned to self.model - it should be the first line in the Initialize function. Note: sometimes QuantConnect is unable to load the model if the project is inactive for a long period. Refresh your page if this problem occurs.

- These strategies can be used for live trading if you have an available live trading node and a Binance account with sufficient funds (although this is not checked so make sure to check manually). Simply initiate live trading and follow the prompts by QuantConnect.

### LSTM Closing Price Prediction With Half Sell Trailing Stop Loss Strategy

This strategy is a simple buy and sell algorithm. It leverages an LSTM model that uses hourly data of the past day (sequence = 24). When the prediction is higher than the current price by 0.5%, the model buys the cryptocurrency. The strategy has a fixed buying allocation of 1/5 of the starting equity per position opened. Once a cryptocurrency dips 1.5% below a trailing stop loss, the strategy sells half of the assets available of the cryptocurrency. Its trading universe is predefined to 10 cryptocurrencies (defined in self.symbols_a).

![Results of Strategy](QC%20Strats/lstm_closing_price_trailing_half/lstm_closing_price_trailing_half%20-%20Equity.png)

A benchmark for the backtesting period is set to buying all the target cryptocurrencies at the start and holding until the end.

![Benchmark of Strategy](QC%20Strats/lstm_closing_price_trailing_half/lstm_closing_price_trailing_half%20-%20Benchmark.png)

### Gradient Boosting Movement Prediction With Half Sell Trailing Stop Loss Strategy

This strategy is similar to the previous one, however, it leverages a gradient boosting model that uses last hour's data to predict the current hour's movement in price. Every hour, it buys cryptocurrencies that are predicted to have an upwards movement. When buying, all available funds are used to open a position and are split equally among the cryptocurrencies to buy. Once a cryptocurrency dips 1.5% below a trailing stop loss, the strategy sells half of the assets available of the cryptocurrency. Its trading universe is predefined to 10 cryptocurrencies (defined in self.symbols_a).

![Results of Strategy](QC%20Strats/portfolio_movement_trailing_half/portfolio_movement_trailing_half%20-%20Equity.png)

A benchmark for the backtesting period is set to buying all the target cryptocurrencies at the start and holding until the end.

![Benchmark of Strategy](QC%20Strats/portfolio_movement_trailing_half/portfolio_movement_trailing_half%20-%20Benchmark.png)

### Note

Please note that there seems to be some error with the QuantConnect backtesting feature causing the results to be inflated. The strategies provided do not use any external sources (except for the models which are trained on data outside of the backtesting period) and thus currently the issue has not been conclusively identified. Use the strategies as reference for how to integrate the models into QuantConnect and as a comparison for what strategies work better than others.