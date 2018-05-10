import tensorflow as tf
import pandas as pd


def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def network(
        data_n, features, batch_size, dataset_train_length,
        num_units, learning_rate, epochs):
    number_of_features = len(features)

    data_length = len(data_n.index) - (len(data_n.index) % batch_size)

    # 1) Prepare datasets

    dataset_train_x = data_n[features].as_matrix()[:dataset_train_length]
    dataset_train_y = data_n['close'].as_matrix()[:dataset_train_length]

    dataset_test_x = data_n[features].as_matrix()[dataset_train_length:]
    dataset_test_y = data_n['close'].as_matrix()[dataset_train_length:]

    # 2) *** Build the network ***

    plh_batch_x = tf.placeholder(
        dtype=tf.float32, name='plc_batch_x',
        shape=[None, batch_size, number_of_features],
    )

    plh_batch_y = tf.placeholder(
        dtype=tf.float32, shape=[None, batch_size, 1], name='plc_batch_x'
    )

    labels_series = tf.unstack(plh_batch_y, axis=1)

    cell = tf.contrib.rnn.BasicRNNCell(num_units=num_units)

    states_series, current_state = tf.nn.dynamic_rnn(
        cell=cell, inputs=plh_batch_x, dtype=tf.float32)
    states_series = tf.transpose(states_series, [1, 0, 2])

    last_state = tf.gather(
        params=states_series, indices=states_series.get_shape()[0] - 1)
    last_label = tf.gather(
        params=labels_series, indices=len(labels_series) - 1)

    weight = tf.Variable(tf.truncated_normal([num_units, 1]))
    bias = tf.Variable(tf.constant(0.1, shape=[1]))

    prediction = tf.matmul(last_state, weight) + bias

    loss = tf.reduce_mean(tf.squared_difference(last_label, prediction))

    train_step = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss)

    l_loss = []
    predicted_data = []

    # 3) *** Start the session ***

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 4) Train the network

        for i_epochs in range(epochs):

            for i_batch in range(dataset_train_length / batch_size):
                i_batch_start = i_batch * batch_size
                i_batch_end = i_batch_start + batch_size

                x = dataset_train_x[
                    i_batch_start:i_batch_end, :].reshape(
                    1, batch_size, number_of_features)
                y = dataset_train_y[i_batch_start:i_batch_end].reshape(
                    1, batch_size, 1)

                feed = {plh_batch_x: x, plh_batch_y: y}

                _loss, _train_step, _pred, _last_label, _prediction = sess.run(
                    fetches=[loss, train_step, prediction, last_label, prediction],
                    feed_dict=feed)

                l_loss.append(_loss)

                # if i_batch % 100 == 0:
                # print('Batch: {} ({}-{}), loss: {}'.format(
                # i_batch, i_batch_start, i_batch_end, _loss))

            print('Epoch: {}, Loss: {}'.format(i_epochs, truncate(l_loss[-1], 8)))

        # 5) Test the Network

        for i_test in range(data_length - dataset_train_length - batch_size):
            # if i_batch % 30 == 0:
            # print('Test: {} ({}-{})'.format(i_test, i_test, i_test + batch_size))

            x = dataset_test_x[
                i_test:i_test + batch_size, :].reshape(
                (1, batch_size, number_of_features))
            y = dataset_test_y[
                i_test:i_test + batch_size].reshape(
                (1, batch_size, 1))

            feed = {plh_batch_x: x, plh_batch_y: y}

            _last_state, _last_label, test_pred = sess.run(
                [last_state, last_label, prediction], feed_dict=feed)
            predicted_data.append(test_pred[-1][0])  # The last one

    # predicted_data = [x for x in predicted_data if str(x) != 'nan']
    predicted_data.extend(
        [None] * (len(data_n) - dataset_train_length - len(predicted_data)))
    df = pd.DataFrame(predicted_data, columns=['predicted'])

    return df.set_index(data_n.index[dataset_train_length:])
