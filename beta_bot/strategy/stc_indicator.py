import talib.abstract as ta
import numpy as np
import pandas as pd


def calculateSTCIndicator(df, length, fastLength, slowLength):
    dataframe = df.copy()
    EEEEEE = length
    BBBB = fastLength
    BBBBB = slowLength
    # dataframe = heikinashi(dataframe)
    mAAAAA = AAAAA(dataframe, EEEEEE, BBBB, BBBBB)
    return mAAAAA


def AAAA(BBB, BBBB, BBBBB):
    fastMA = ta.EMA(BBB, timeperiod=BBBB)
    slowMA = ta.EMA(BBB, timeperiod=BBBBB)
    AAAA = fastMA - slowMA
    return AAAA


def AAAAA(dataframe, EEEEEE, BBBB, BBBBB):

    AAA = 0.5
    dataframe['DDD'] = 0.0
    dataframe['CCCCC'] = 0
    dataframe['DDDDDD'] = 0
    dataframe['EEEEE'] = 0.0

    dataframe['BBBBBB'] = AAAA(dataframe['close'], BBBB, BBBBB)
    dataframe['CCC'] = dataframe['BBBBBB'].rolling(window=EEEEEE).min()
    dataframe['CCCC'] = dataframe['BBBBBB'].rolling(
        window=EEEEEE).max() - dataframe['CCC']
    dataframe['CCCCC'] = np.where(dataframe['CCCC'] > 0, (dataframe['BBBBBB'] -
                                  dataframe['CCC']) / dataframe['CCCC'] * 100, dataframe['CCCCC'].shift(1).fillna(0))

    for i in range(0, len(dataframe)):
        if (i > 0):
            dataframe.at[i, 'DDD'] = dataframe['DDD'].iloc[i-1] + \
                (AAA * (dataframe.at[i, 'CCCCC'] - dataframe['DDD'].iloc[i-1]))

    dataframe['DDDD'] = dataframe['DDD'].rolling(window=EEEEEE).min()
    dataframe['DDDDD'] = dataframe['DDD'].rolling(
        window=EEEEEE).max() - dataframe['DDDD']
    dataframe['DDDDDD'] = np.where(dataframe['DDD'] > 0, (dataframe['DDD'] - dataframe['DDDD']) /
                                   dataframe['DDDDD'] * 100, dataframe['DDD'].fillna(dataframe['DDDDDD'].shift(1)))

    for i in range(0, len(dataframe)):
        if (i > 0):
            dataframe.at[i, 'EEEEE'] = dataframe['EEEEE'].iloc[i-1] + \
                (AAA * (dataframe.at[i, 'DDDDDD'] -
                 dataframe['EEEEE'].iloc[i-1]))

    dataframe.loc[
        (
            (dataframe['EEEEE'] > dataframe['EEEEE'].shift(1))
        ),
        'stc_signal'] = 1

    dataframe.loc[
        (
            (dataframe['EEEEE'] < dataframe['EEEEE'].shift(1))
        ),
        'stc_signal'] = -1

    dataframe.loc[
        (
            (dataframe['stc_signal'] == 1)
            &
            (dataframe['stc_signal'].shift(1) == -1)
        ),
        'stc_entry_exit'] = 1

    dataframe.loc[
        (
            (dataframe['stc_signal'] == -1)
            &
            (dataframe['stc_signal'].shift(1) == 1)
        ),
        'stc_entry_exit'] = -1

    return dataframe['stc_signal']
