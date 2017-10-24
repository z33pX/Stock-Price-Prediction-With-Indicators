# Stock-Price-Prediction-With-Indicators

This program is for testing and experimenting with indicators to predict prices of stocks.
The prediction is based on a recurrent neural network. You can find all following listed variables in `main.py`

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

* Batch size: `batch_size = 3`
* Size of the test dataset. 0.1 means 10% of the complete dataset. The remaining 90% are testing data: `test_dataset_size = 0.1`
* Neurons of the RNN: `num_units = 12`
* Learning rate: `learning_rate = 0.001`
* Epochs: `epochs = 10`

Try different values for different stocks to get better results. The parameters above worked fine for me for Tesla (`'TSLA'`)

Data
-

The dataset is specifiable by a ticker symbol like `'TSLA'` (Tesla), `'MSFT'` (Microsoft) or `'AMZN'` 
(Amazon). The first time you are starting the program the data will be downloaded from Yahoo Finance and stored in a pickle file.
The size of the dataset can be adjusted by  `start_date` and `end_date`.

```
ticker = 'TSLA'
end_date = str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d'))

df = get_data(ticker=ticker, start_date='20000101', end_date=end_date)
```
The first part of the dataset will be cut by the length of the longest moving average to prevent empty data.

Graph
-

All indicators can be visualised by setting the corresponding variable

```
draw_ATR=True
draw_MACD=True
draw_Stochastics=True
draw_RSI=True

# By setting the labels of the ma the ma will be visible in the graph
moving_average_1 = ci.moving_average_1_label
moving_average_2 = None
```

Result
-

Below you can see an example result of the parameter configuration described on this page.

![](https://github.com/z33pX/Stock-Price-Prediction-With-Indicators/blob/master/pic_1.png)

Another result plus the graphs of all indicators:

![](https://github.com/z33pX/Stock-Price-Prediction-With-Indicators/blob/master/pic_2.png)
