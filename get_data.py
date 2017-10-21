import pandas as pd
import pandas_datareader.data as web
from pathlib import Path as pl


def get_data(ticker, start_date, end_date):

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

    return main_df
    