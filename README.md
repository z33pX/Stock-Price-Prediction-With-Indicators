# Stock-Price-Prediction-With-Indicators

This program is for experimenting and testing with indicators of the technical analysis for predicting prices of stocks.
The prediction is based on a recurrent neural network.

Indicators
-

The following indicators are available as features:

* Close (of the day before) -- `'Close'`
* MACD  -- `'MACD'`
* Stochastic Oscillator -- `'Stochastics'`
* ATR (Average True Range) -- `'ATR'`
* RSI (Relative Strength Index) -- `'RSI'`
* Moving Average 1 -- `'ci.moving_average_1_label'`
* Moving Average 2 -- `'ci.moving_average_2_label'`

You can add or remove these indicators by editing the feature list:

```
features = ['MACD', ci.moving_average_1_label]
```

Parameters of indicators
-

The parameters of the indicator are alterable. Try different values and combinations of indicators as features to reduce the tracking error.

```
ci.set_RSI_parameter(n=14)
ci.set_MACD_parameter(fast=12, slow=26, signal=9)
ci.set_SO_parameter(period=14)
ci.set_moving_average_1(window=12)
ci.set_moving_average_2(window=26)
```

Network parameters
-

* Batch size: `'batch_size = 3'`
* Size of the test dataset. 0.1 means 10% of the complete dataset. The remaining 90% are testing data: `'test_dataset_size = 0.1'`
* Neurons of the RNN: `'num_units = 12'`
* Learning rate: `'learning_rate = 0.001'`
* Epochs: `'epochs = 10'`

Try different values for different stocks to get better results. The parameters above worked fine for me for Tesla (`'TSLA'`)

Data
-

The dataset will be downloaded by a specified ticker symbol like `'TSLA'` (Tesla), `'MSFT'` (Microsoft) or `'AMZN'` 
(Amazon) from Yahoo Finance. The size of the dataset can be adjusted by the `'start_date'` and the `'end_date'`.

```
ticker = 'TSLA'
end_date = str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d'))

df = get_data(ticker=ticker, start_date='20000101', end_date=end_date)
```
The first part of the dataset will be cut by the length of the longest moving average to prevent empty data.

Result
-

Below you can see an example result of the parameter configuration described on this page.

![](https://github.com/z33pX/Stock-Price-Prediction-With-Indicators/blob/master/pic.png)
