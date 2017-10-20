import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.transforms as mtrans
from matplotlib.patches import BoxStyle
import pandas_datareader.data as web
from pathlib import Path as pl
import datetime
import time

# Disable UserWarning: export TF_CPP_MIN_LOG_LEVEL=2


def get_date():
    return str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d'))


def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


class MyStyle(BoxStyle._Base):

    def __init__(self, pad=0.3):
        self.pad = pad
        super(MyStyle, self).__init__()

    def transmute(self, x0, y0, width, height, mutation_size):

        # padding
        pad = mutation_size * self.pad

        # width and height with padding added.
        width, height = width + 2.*pad, height + 2.*pad,

        # boundary of the padded box
        # x0 = x0 + 3 + width / 2
        x0, y0 = x0-pad, y0-pad,
        x1, y1 = x0+width, y0 + height

        cp = [(x0, y0),
              (x1, y0), (x1, y1), (x0, y1),
              (x0-pad, (y0+y1)/2.), (x0, y0),
              (x0, y0)]

        com = [Path.MOVETO,
               Path.LINETO, Path.LINETO, Path.LINETO,
               Path.LINETO, Path.LINETO,
               Path.CLOSEPOLY]

        path = Path(cp, com)

        return path


def rsi_function(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n + 1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    relative_strengh = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + relative_strengh)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (n - 1) + upval)/n
        down = (down * (n - 1) + downval)/n

        relative_strengh = up/down
        rsi[i] = 100. - 100. / (1. + relative_strengh)

    return rsi


def stochastics_oscillator(df, period):
    l, h = pd.DataFrame.rolling(df, period).min(), pd.DataFrame.rolling(df, period).max()
    return 100 * (df - l) / (h - l)


def MACD(df, period_1, period_2, period_signal):
    ema1 = pd.DataFrame.ewm(df, span=period_1).mean()
    ema2 = pd.DataFrame.ewm(df, span=period_2).mean()
    MACD = ema1 - ema2
    
    signal = pd.DataFrame.ewm(MACD, period_signal).mean()
    
    return MACD - signal


def ATR(df, period):
    df['H-L'] = abs(df['High'] - df['Low'])
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    return df[['H-L', 'H-PC', 'L-PC']].max(axis=1).to_frame()

## *********************************************************************************
## 0) *** Download data ***

# Tesla:
ticker = 'TSLA'

# SP500:
# ticker = '^GSPC'

start_date = '20000101'
end_date = get_date()
file_path = ticker + '_data.ple'

file = pl(file_path)

if file.exists():
    print('load data from file ...')
    main_df = pd.read_pickle(file_path)

else:
    print('download data ...')
    main_df = web.DataReader(ticker, 'yahoo', start_date, end_date)
    main_df.index.names = ['Date']

    print('save data to file ...')
    main_df.to_pickle(file_path)

## *********************************************************************************
## 1) *** Calculate indicators ***

# (exponential) moving averages:
moving_average_1_window = 100
moving_average_1 = 'ma_' + str(moving_average_1_window)
moving_average_2_window = 200
moving_average_2 = 'ma_' + str(moving_average_2_window)

atr_df = ATR(main_df, 14)
atr_df.rename(columns={0:'ATR'}, inplace=True)

main_df = pd.DataFrame(main_df['Close'])
main_df = main_df.iloc[::-1]

moving_average_1_df = main_df.rolling(window=moving_average_1_window, center=False).mean().dropna()
z = np.zeros(len(main_df.index) - len(moving_average_1_df.index))
moving_average_1_df = pd.DataFrame(np.append(moving_average_1_df['Close'].values, z), 
    index=main_df.index, columns={moving_average_1})

moving_average_2_df = main_df.rolling(window=moving_average_2_window, center=False).mean().dropna()
z = np.zeros(len(main_df.index) - len(moving_average_2_df.index))
moving_average_2_df = pd.DataFrame(np.append(moving_average_2_df['Close'].values, z), 
    index=main_df.index, columns={moving_average_2})

rsi = rsi_function(main_df['Close'])
rsi_df = pd.DataFrame(rsi, index=main_df.index, columns={'RSI'})

stochastics_df = stochastics_oscillator(main_df,14)
stochastics_df.rename(columns={'Close':'Stochastics'}, inplace=True)

macd_df = MACD(main_df, 12, 26, 9)
macd_df.rename(columns={'Close':'MACD'}, inplace=True)

close_target_df = main_df.shift(-1)
close_target_df.rename(columns={'Close':'CloseTarget'}, inplace=True)

data = pd.concat([main_df, close_target_df, macd_df, rsi_df, moving_average_1_df, 
    moving_average_2_df, stochastics_df, atr_df], axis=1)
data = data.iloc[moving_average_2_window:]
data = data.dropna()

## *********************************************************************************
## 2) *** Set parameters and prepare train and test datasets ***

# Parameters
batch_size = 3
test_dataset_size = 0.1 # = 10 percent of the complete dataset for testing
num_units = 12
learning_rate = 0.001
epochs = 10

# All features, which can be used for training:
# ['Close', 'MACD', 'Stochastics', 'ATR', 'RSI', moving_average_1, moving_average_2]
features = ['Close', 'MACD', 'Stochastics', 'ATR']

data_length = len(data.index) - (len(data.index) % batch_size)
data = (data - data.mean()) / (data.max() - data.min())[:data_length]

dataset_train_length = data_length - int(len(data.index) * test_dataset_size)

dataset_train_x = data[features].as_matrix()[:dataset_train_length]
dataset_train_y = data['CloseTarget'].as_matrix()[:dataset_train_length]

dataset_test_x = data[features].as_matrix()[dataset_train_length:]
dataset_test_y = data['CloseTarget'].as_matrix()[dataset_train_length:]

number_of_features = len(features)

## *********************************************************************************
## 3) *** Build the network ***

plh_batch_x = tf.placeholder(dtype=tf.float32,
                             shape=[None, batch_size, number_of_features], name='plc_batch_x')

plh_batch_y = tf.placeholder(dtype=tf.float32,
                             shape=[None, batch_size, 1], name='plc_batch_x')

labels_series = tf.unstack(plh_batch_y, axis=1)

cell = tf.contrib.rnn.BasicRNNCell(num_units=num_units)

states_series, current_state = tf.nn.dynamic_rnn(cell=cell, inputs=plh_batch_x, dtype=tf.float32)
states_series = tf.transpose(states_series, [1, 0, 2])

last_state = tf.gather(params=states_series, indices=states_series.get_shape()[0] - 1)
last_label = tf.gather(params=labels_series, indices=len(labels_series) - 1)

weight = tf.Variable(tf.truncated_normal([num_units, 1]))
bias = tf.Variable(tf.constant(0.1, shape=[1]))

prediction = tf.matmul(last_state, weight) + bias

loss = tf.reduce_mean(tf.squared_difference(last_label, prediction))

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

l_loss = []
l_test_pred = []

## *********************************************************************************
## 4) *** Start the session ***

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    ## 5) Train the network

    for i_epochs in range(epochs):
        print('Epoch: {}'.format(i_epochs))

        for i_batch in range(dataset_train_length / batch_size):
            i_batch_start = i_batch * batch_size
            i_batch_end = i_batch_start + batch_size

            x = dataset_train_x[i_batch_start:i_batch_end, :].reshape(1, batch_size, number_of_features)
            y = dataset_train_y[i_batch_start:i_batch_end].reshape(1, batch_size, 1)

            feed = {plh_batch_x: x, plh_batch_y: y}

            _loss, _train_step, _pred, _last_label, _prediction = sess.run(
                            fetches=[loss, train_step, prediction, last_label, prediction],
                            feed_dict=feed)

            l_loss.append(_loss)

            # if i_batch % 100 == 0:
                # print('Batch: {} ({}-{}), loss: {}'.format(i_batch, i_batch_start, i_batch_end, _loss))

    ## 6) Test the Network

    for i_test in range(data_length - dataset_train_length - batch_size):

        # if i_batch % 30 == 0:
            # print('Test: {} ({}-{})'.format(i_test, i_test, i_test + batch_size))

        x = dataset_test_x[i_test:i_test + batch_size, :].reshape((1, batch_size, number_of_features))
        y = dataset_test_y[i_test:i_test + batch_size].reshape((1, batch_size, 1))

        feed = {plh_batch_x: x, plh_batch_y: y}

        _last_state, _last_label, test_pred = sess.run([last_state, last_label, prediction], feed_dict=feed)
        l_test_pred.append(test_pred[-1][0])  # The last one

## *********************************************************************************
## 7) Draw graph

BoxStyle._style_list["angled"] = MyStyle

fig = plt.figure(facecolor='#000606')
plt.subplots_adjust(left=.08, bottom=.08, right=.96, top=.96, hspace=.0, wspace=.06)
plt.suptitle(ticker, color='#00decc')

ax_price = plt.subplot2grid((4, 4), (0, 0), rowspan=4, colspan=4, facecolor='#000606')
trans_offset = mtrans.offset_copy(ax_price.transData, fig=fig, x=0.15, y=0.0, units='inches')

# Plot test dataset
ax_price.plot(dataset_test_y, label='Price', color='#f600ff', linewidth=0.7)
ax_price.text(len(dataset_test_y) - 1, dataset_test_y[-1], truncate(dataset_test_y[-1], 2),
              size=7, va="center", ha="center", transform=trans_offset,
              bbox=dict(boxstyle="angled,pad=0.2", alpha=0.6, color='#f600ff'))

# Plot predicted data
ax_price.plot(l_test_pred, 'r--', label='Predicted', color='#ffba00', linewidth=0.7)
l_test_pred = [x for x in l_test_pred if str(x) != 'nan']
ax_price.text(len(l_test_pred) - 1, l_test_pred[-1], truncate(l_test_pred[-1], 2),
              size=7, va="center", ha="center", transform=trans_offset,
              bbox=dict(boxstyle="angled,pad=0.2", alpha=0.6, color='#ffba00'))

# Add tracking error box
tracking_error = truncate(np.std(l_test_pred - dataset_test_y[:len(l_test_pred)]) * 100, 2)
ax_price.annotate('Tracking Error: ' + str(tracking_error) + '%', xy=(0.7, 0.05),
                  xycoords='axes fraction', fontsize=10, bbox=dict(facecolor='#ffba00', alpha=0.6),
                  ha='left', va='bottom')

ax_price.grid(linestyle='dotted')
ax_price.yaxis.label.set_color('#037f7a')
ax_price.legend(loc='upper left')
ax_price.spines['left'].set_color('#037f7a')
ax_price.spines['right'].set_color('#000606')
ax_price.spines['top'].set_color('#000606')
ax_price.spines['bottom'].set_color('#000606')
ax_price.tick_params(axis='y', colors='#037f7a')
ax_price.tick_params(axis='x', colors='#037f7a')

ax_price.fill_between(np.arange(0, len(l_test_pred), 1), dataset_test_y[:len(l_test_pred)], l_test_pred,
                      alpha=.05, color='#ffba00')

legend = ax_price.legend(loc='best', fancybox=True, framealpha=0.5)
legend.get_frame().set_facecolor('#000606')
for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.show()
