import tensorflow as tf
from graph import draw
from graph import show
from graph import save
from get_data import get_data
from indicators import CalculateIndicators
import datetime
import time

# Disable UserWarnings on linux promt: export TF_CPP_MIN_LOG_LEVEL=2

def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

## *********************************************************************************
## 0) *** Download data ***

ticker = 'TSLA'
end_date = str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d'))

df = get_data(ticker=ticker, start_date='20000101', end_date=end_date)

## *********************************************************************************
## 1) *** Calculat indicators ***

ci = CalculateIndicators(df)

# Parameters
ci.set_RSI_parameter(n=14)
ci.set_MACD_parameter(fast=12, slow=26, signal=9)
ci.set_SO_parameter(period=14)
ci.set_moving_average_1(window=12)
ci.set_moving_average_2(window=26)

data = ci.calculate_indicators()

## *********************************************************************************
## 2) *** Set parameters and prepare datasets for training and testing ***

# Parameters
batch_size = 3
test_dataset_size = 0.1  # = 10 percent of the complete dataset for testing
num_units = 12
learning_rate = 0.001
epochs = 10

# All available features:
# ['Close', 'MACD', 'Stochastics', 'ATR', 'RSI', ci.moving_average_1_label, ci.moving_average_2_label]
features = ['MACD', ci.moving_average_1_label]
number_of_features = len(features)

data_length = len(data.index) - (len(data.index) % batch_size)
data_n = (data - data.mean()) / (data.max() - data.min())[:data_length]

dataset_train_length = data_length - int(len(data_n.index) * test_dataset_size)

dataset_train_x = data_n[features].as_matrix()[:dataset_train_length]
dataset_train_y = data_n['CloseTarget'].as_matrix()[:dataset_train_length]

dataset_test_x = data_n[features].as_matrix()[dataset_train_length:]
dataset_test_y = data_n['CloseTarget'].as_matrix()[dataset_train_length:]

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
predicted_data = []

## *********************************************************************************
## 4) *** Start the session ***

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    ## 5) Train the network

    for i_epochs in range(epochs):

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

        print('Epoch: {}, Loss: {}'.format(i_epochs, truncate(l_loss[-1], 8)))

    ## 6) Test the Network

    for i_test in range(data_length - dataset_train_length - batch_size):

        # if i_batch % 30 == 0:
            # print('Test: {} ({}-{})'.format(i_test, i_test, i_test + batch_size))

        x = dataset_test_x[i_test:i_test + batch_size, :].reshape((1, batch_size, number_of_features))
        y = dataset_test_y[i_test:i_test + batch_size].reshape((1, batch_size, 1))

        feed = {plh_batch_x: x, plh_batch_y: y}

        _last_state, _last_label, test_pred = sess.run([last_state, last_label, prediction], feed_dict=feed)
        predicted_data.append(test_pred[-1][0])  # The last one

## *********************************************************************************
## 7) Draw graph

# Parameters
draw_ATR=True
draw_MACD=True
draw_Stochastics=True
draw_RSI=True

# By setting the labels of the ma the ma will be visible in the graph
moving_average_1 = ci.moving_average_1_label
moving_average_2 = None

# I love to play around with colors :)
accent_color = '#c9c9c9'
indicators_color = '#007b9f'

# The use of rescaled data is necessary for plotting the price and moving averages in the same graph.
data['Close'] = data_n['Close']
data[ci.moving_average_1_label] = data_n[ci.moving_average_1_label]
data[ci.moving_average_2_label] = data_n[ci.moving_average_2_label]

# Draw
draw(ticker, data[dataset_train_length:], predicted_data, moving_average_1, moving_average_2,
     draw_ATR=draw_ATR, draw_MACD=draw_MACD, draw_Stochastics=draw_Stochastics, draw_RSI=draw_RSI,
     accent_color=accent_color, indicators_color=indicators_color)

show()
# save('graph.png')
