# Stock-Price-Prediction-With-Indicators

This program is for testing and experimenting with indicators to predict prices of stocks or cryptocurrencies.
The prediction is based on a recurrent neural network. To make useful predictions based on this approach it is necessary 
to predict the indicators first. **The goal of this program is not to make final predictions to decide whether to invest or not**. 
But it's purpose is to experiment with indicators on the RNN approach to find more or less useful 
indicators and configurations.

The dataset in this example is a `csv` file of 5min BTC_XRP `open`, `high`, `low`, `close` ... data.
It provides more than 120000 datapoints. As features for the network we will use indicators. 
The file `indicators.py` provides many indicators copied from the
[**pandas-technical-indicators**](https://github.com/Crypto-toolbox/pandas-technical-indicators)
repository from [**Crypto-toolbox**](https://github.com/Crypto-toolbox). Some of them are modified because
the index did't work for me. Not all indicators are tested so maybe some of them have to be fixed first.
The net we're using after preparing data is a recurrent neural network. At the end  we'll plot everything with
[**mpl_finance_ext**](https://github.com/z33pX/mpl_finance_ext) (outdated! Please follow link for current version).

Dataset and features
-

The first step is to prepare the data. We load the dataset and calculate the indicators. The dataset
was downloaded from the [**Poloniex**](https://poloniex.com/) api. The indicators are copied from the
[**pandas-technical-indicators**](https://github.com/Crypto-toolbox/pandas-technical-indicators)
repository from [**Crypto-toolbox**](https://github.com/Crypto-toolbox) as mentioned earlier.

```
# Load dataset
data = pd.read_csv('BTC_XRP_5min.csv', index_col=0).tail(5000)
data = data.drop(['date', 'quoteVolume', 'volume', 'weightedAverage'], 1)

# Calculate indicators
data = relative_strength_index(df=data, n=14)
data = bollinger_bands(df=data, n=20, std=4, add_ave=False)
data = exponential_moving_average(df=data, n=10)
data = moving_average(df=data, n=12)
data = macd(df=data, n_fast=12, n_slow=26)

# Cut DataFrame
data = data.iloc[40::]
# Reset index
data = data.reset_index()
# Delete old index
data = data.drop('index', 1)

# Normalize data
data_n = (data - data.mean()) / (data.max() - data.min())
```
To select a feature just enter the name of the column to the feature list:

```
features = ['MA_12', 'MACD_12_26']
```
To find out which names are existing just add `print(list(data_n))`. In this example available are
`['close', 'high', 'low', 'open', 'RSI_14', 'bband_upper_20', 'bband_lower_20', 
'EMA_10', 'MA_12', 'MACD_12_26', 'MACDsign_12_26', 'MACDdiff_12_26']`

Network parameters
-

* Batch size: `batch_size = 3`
* Size of the test dataset. 0.1 means 10% of the complete dataset. The remaining 90% are testing data: `test_dataset_size = 0.1`
* Neurons of the RNN: `num_units = 12`
* Learning rate: `learning_rate = 0.001`
* Epochs: `epochs = 8`

Plot everything
-

For plotting we are using [**mpl_finance_ext**](https://github.com/z33pX/mpl_finance_ext) from another Github repository 
of mine. The project provides a documentation.

Result
-

Below you can see an example result of the parameter configuration described on this page.
The white area shows the training data and the other area on the right in the upper chart shows 
the predicted data.

![](https://github.com/z33pX/Stock-Price-Prediction-With-Indicators/blob/master/pic_01.png)

Here we take a closer look:

![](https://github.com/z33pX/Stock-Price-Prediction-With-Indicators/blob/master/pic_02.png)
