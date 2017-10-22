import numpy as np
import pandas as pd


class CalculateIndicators(object):

    df = None

    RSI_n = 14

    stochastic_oscillator_period = 14

    MACD_period_1 = 12
    MACD_period_2 = 26
    MACD_period_signal = 9

    moving_average_1_value = 12
    moving_average_2_value = 26

    moving_average_1_label = 'ma_' + str(moving_average_1_value)
    moving_average_2_label = 'ma_' + str(moving_average_2_value)

    def __init__(self, df):
        self.df = df

    def set_RSI_parameter(self, n):
        self.RSI_n = n

    def set_MACD_parameter(self, period_1, period_2, period_signal):
        self.MACD_period_1 = period_1
        self.MACD_period_2 = period_2
        self.MACD_period_signal = period_signal

    def set_SO_parameter(self, period):
        self.stochastic_oscillator_period = period

    def set_moving_average_1(self, window):
        self.moving_average_1_value = window
        self.moving_average_1_label = 'ma_' + str(self.moving_average_1_value)

    def set_moving_average_2(self, window):
        self.moving_average_2_value = window
        self.moving_average_2_label = 'ma_' + str(self.moving_average_2_value)


    def RSI(self, n):
        deltas = np.diff(self.df['Close'])
        seed = deltas[:n + 1]
        up = seed[seed >= 0].sum() / n
        down = -seed[seed < 0].sum() / n
        relative_strengh = up / down
        rsi = np.zeros_like(self.df['Close'])
        rsi[:n] = 100. - 100. / (1. + relative_strengh)

        for i in range(n, len(self.df['Close'])):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (n - 1) + upval) / n
            down = (down * (n - 1) + downval) / n

            relative_strengh = up / down
            rsi[i] = 100. - 100. / (1. + relative_strengh)

        return pd.DataFrame(rsi, index=self.df.index, columns={'RSI'})

    def stochastic_oscillator(self, period):
        close = self.df['Close'].to_frame()
        l = pd.DataFrame.rolling(close, period).min()
        h = pd.DataFrame.rolling(close, period).max()

        _df = 100 * (close - l) / (h - l)
        _df.rename(columns={'Close':'Stochastics'}, inplace=True)

        return _df

    def MACD(self, fast, slow, signal):
        close = self.df['Close'].to_frame()
        ema1 = pd.DataFrame.ewm(close, span=fast).mean()
        ema2 = pd.DataFrame.ewm(close, span=slow).mean()
        MACD = ema1 - ema2

        signal = pd.DataFrame.ewm(MACD, signal).mean()

        _df = (MACD - signal)
        _df.rename(columns={'Close':'MACD'}, inplace=True)

        return _df

    def ATR(self, period):
        _df = pd.DataFrame()

        _df['H-L'] = abs(self.df['High'] - self.df['Low'])
        _df['H-PC'] = abs(self.df['High'] - self.df['Close'].shift(1))
        _df['L-PC'] = abs(self.df['Low'] - self.df['Close'].shift(1))
        _df = pd.DataFrame(_df[['H-L', 'H-PC', 'L-PC']].max(axis=1), columns={'ATR'})

        return _df


    def MA(self, value, column):
        _df = self.df['Close'].rolling(window=value, center=False).mean().dropna()
        z = np.full(len(self.df.index) - len(_df.index), np.nan)

        return pd.DataFrame(np.append(z, _df.values), index=self.df.index, columns={column})


    def calculate_indicators(self):

        atr_df = self.ATR(14)

        df = pd.DataFrame(self.df['Close'])

        ma_1_df = self.MA(self.moving_average_1_value, self.moving_average_1_label)

        ma_2_df = self.MA(self.moving_average_2_value, self.moving_average_2_label)

        rsi_df = self.RSI(self.RSI_n)

        stochastics_df = self.stochastic_oscillator(self.stochastic_oscillator_period)

        macd_df = self.MACD(self.MACD_period_1, self.MACD_period_2, self.MACD_period_signal)

        close_target_df = df.shift(-1)
        close_target_df.rename(columns={'Close':'CloseTarget'}, inplace=True)

        data = pd.concat([df, close_target_df, macd_df, rsi_df, ma_1_df, ma_2_df, stochastics_df, atr_df], axis=1)

        return data.dropna()
