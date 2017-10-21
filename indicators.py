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


    def RSI(self, prices, n):
        deltas = np.diff(prices)
        seed = deltas[:n + 1]
        up = seed[seed >= 0].sum() / n
        down = -seed[seed < 0].sum() / n
        relative_strengh = up / down
        rsi = np.zeros_like(prices)
        rsi[:n] = 100. - 100. / (1. + relative_strengh)

        for i in range(n, len(prices)):
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

        return rsi

    def stochastic_oscillator(self, period):
        l = pd.DataFrame.rolling(self.df, period).min()
        h = pd.DataFrame.rolling(self.df, period).max()
        return 100 * (self.df - l) / (h - l)

    def MACD(self, period_1, period_2, period_signal):
        ema1 = pd.DataFrame.ewm(self.df, span=period_1).mean()
        ema2 = pd.DataFrame.ewm(self.df, span=period_2).mean()
        MACD = ema1 - ema2

        signal = pd.DataFrame.ewm(MACD, period_signal).mean()

        return MACD - signal

    def ATR(self, period):
        df = self.df
        df['H-L'] = abs(self.df['High'] - self.df['Low'])
        df['H-PC'] = abs(self.df['High'] - self.df['Close'].shift(1))
        df['L-PC'] = abs(self.df['Low'] - self.df['Close'].shift(1))
        return df[['H-L', 'H-PC', 'L-PC']].max(axis=1).to_frame()

    def calculate_indicators(self):

        atr_df = self.ATR(14)
        atr_df.rename(columns={0:'ATR'}, inplace=True)

        df = pd.DataFrame(self.df['Close'])
        df = df.iloc[::-1]

        ma_1_df = df.rolling(window=self.moving_average_1_value, center=False).mean().dropna()
        z = np.zeros(len(df.index) - len(ma_1_df.index))
        ma_1_df = pd.DataFrame(np.append(ma_1_df['Close'].values, z), index=df.index, columns={self.moving_average_1_label})

        ma_2_df = df.rolling(window= self.moving_average_2_value, center=False).mean().dropna()
        z = np.zeros(len(df.index) - len(ma_2_df.index))
        ma_2_df = pd.DataFrame(np.append(ma_2_df['Close'].values, z), index=df.index, columns={self.moving_average_2_label})

        rsi = self.RSI(df['Close'], self.RSI_n)
        rsi_df = pd.DataFrame(rsi, index=df.index, columns={'RSI'})

        stochastics_df = self.stochastic_oscillator(self.stochastic_oscillator_period)
        stochastics_df.rename(columns={'Close':'Stochastics'}, inplace=True)

        macd_df = self.MACD(self.MACD_period_1, self.MACD_period_2, self.MACD_period_signal)
        macd_df.rename(columns={'Close':'MACD'}, inplace=True)

        close_target_df = df.shift(-1)
        close_target_df.rename(columns={'Close':'CloseTarget'}, inplace=True)

        data = pd.concat([df, close_target_df, macd_df, rsi_df, ma_1_df, ma_2_df, stochastics_df, atr_df], axis=1)

        if self.moving_average_2_value > self.moving_average_1_value:
            data = data.iloc[self.moving_average_2_value:]
        else:
            data = data.iloc[self.moving_average_1_value:]

        return data.dropna()
