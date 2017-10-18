import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# export TF_CPP_MIN_LOG_LEVEL=2

## 1) *** Prepare data ***

# Data parameters
ticker = 'TSLA'
batch_size = 3
test_dataset_size = 0.1 # = 10 percent of the complete dataset
number_of_features = 4

# Network parameters
num_units = 12
learning_rate = 0.001
epochs = 20

data = pd.read_csv(ticker + '_technical_indicators.csv')
data = data.set_index(['Date'])
data_length = len(data.index) - (len(data.index) % batch_size)
data = (data - data.mean()) / (data.max() - data.min())[:data_length]

dataset_train_length = data_length - int(len(data.index) * test_dataset_size)
dataset_train_x = data[['Close', 'MACD', 'Stochastics', 'ATR']].as_matrix()[:dataset_train_length]
dataset_train_y = data['CloseTarget'].as_matrix()[:dataset_train_length]

dataset_test_x = data[['Close', 'MACD', 'Stochastics', 'ATR']].as_matrix()[dataset_train_length:]
dataset_test_y = data['CloseTarget'].as_matrix()[dataset_train_length:]

## 2) *** Building the network ***

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


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    ## 3) Train the network

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

    ## 4) Test the Network

    for i_test in range(data_length - dataset_train_length - batch_size):

        # if i_batch % 30 == 0:
            # print('Test: {} ({}-{})'.format(i_test, i_test, i_test + batch_size))

        x = dataset_test_x[i_test:i_test + batch_size, :].reshape((1, batch_size, number_of_features))
        y = dataset_test_y[i_test:i_test + batch_size].reshape((1, batch_size, 1))

        feed = {plh_batch_x: x, plh_batch_y: y}

        _last_state, _last_label, test_pred = sess.run([last_state, last_label, prediction], feed_dict=feed)
        l_test_pred.append(test_pred[-1][0])  # The last one


## 5) Draw graph

fig = plt.figure(facecolor='#000606')
plt.suptitle(ticker, color='#00decc')

ax_price = plt.subplot2grid((4, 4), (0, 0), rowspan=4, colspan=4, facecolor='#000606')
ax_price.set_title(ticker)
ax_price.grid(linestyle='dotted')

ax_price.yaxis.label.set_color('#00decc')
ax_price.plot(dataset_test_y, label='Price', color='#00decc', linewidth=0.5)
ax_price.plot(l_test_pred, label='Predicted', color='#f600ff', linewidth=0.5)
ax_price.legend(loc='upper left')
ax_price.spines['bottom'].set_color('#037f7a')
ax_price.spines['left'].set_color('#037f7a')
ax_price.tick_params(axis='y', colors='#037f7a')

plt.show()
