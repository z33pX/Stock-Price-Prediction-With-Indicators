from network import network
from indicators import *
import mpl_finance_ext as mfe
import matplotlib.pyplot as plt


def main():

    # 1) Arrange data -------------------------------------------------
    # Load dataset (max 120000 datapoints)
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

    # 2) RNN ----------------------------------------------------------
    # Parameters
    batch_size = 3
    test_dataset_size = 0.05
    num_units = 12
    learning_rate = 0.001
    epochs = 8

    # Which names are avaiable? print(list(data_n))
    features = ['MA_12', 'MACD_12_26']

    dataset_train_length = len(data_n.index) -\
        int(len(data_n.index) * test_dataset_size)

    training_data = data_n.iloc[:dataset_train_length]

    # Train and test the RNN
    predicted_data = network(
        data_n, features, batch_size,
        dataset_train_length, num_units,
        learning_rate, epochs
    )

    # Append test close data and the predicted data
    test_close = pd.DataFrame(data_n['close'][dataset_train_length::])
    df = pd.concat([training_data, predicted_data, test_close])

    # 3) Plot ---------------------------------------------------------
    fig, _ = plt.subplots(facecolor=mfe.background_color)
    ax0 = plt.subplot2grid(
        (10, 8), (0, 0),
        rowspan=6, colspan=8,
        facecolor=mfe.background_color
    )

    mfe.plot_candlestick(
        fig=fig,
        axis=ax0,
        data=df,
        plot_columns=[
            'bband_upper_20', 'bband_lower_20',
            'MA_12', 'predicted', 'close'
        ],
        vline=dataset_train_length - 1,
        vspan=[dataset_train_length - 1, len(data_n.index)],
    )

    tracking_error = np.std(
        predicted_data['predicted'] -
        data_n['close'][dataset_train_length::]) * 100
    print('Tracking_error: ' + str(tracking_error))

    # Plot RSI
    ax1 = plt.subplot2grid(
        (10, 8), (6, 0),
        rowspan=2, colspan=8, sharex=ax0,
        facecolor=mfe.background_color
    )

    mfe.plot(
        data=data_n,
        name='RSI_14',
        plot_columns=['RSI_14'],
        axis=ax1,
        fig=fig,
        xhline_red=0.8,
        xhline_green=0.2,
        vline=dataset_train_length - 1,
        vspan=[dataset_train_length - 1, len(data_n.index)]
    )

    # Plot MACD
    ax1 = plt.subplot2grid(
        (10, 8), (8, 0),
        rowspan=2, colspan=8, sharex=ax0,
        facecolor=mfe.background_color
    )

    mfe.plot(
        data=data_n,
        name='MACD',
        plot_columns=['MACD_12_26', 'MACDsign_12_26', 'MACDdiff_12_26'],
        axis=ax1,
        fig=fig,
        xhline_dashed1=0,
        vline=dataset_train_length - 1,
        vspan=[dataset_train_length - 1, len(data_n.index)]
    )
    plt.subplots_adjust(
        left=.07, bottom=.05, right=.94,
        top=.96, hspace=0.2, wspace=0.03
    )
    plt.show()


if __name__ == "__main__":
    main()
