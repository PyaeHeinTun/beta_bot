import talib.abstract as ta
import numpy as np
import pandas as pd


def xATRTrailingStop_func(close, prev_close, prev_atr, nloss):
    if close > prev_atr and prev_close > prev_atr:
        return max(prev_atr, close - nloss)
    elif close < prev_atr and prev_close < prev_atr:
        return min(prev_atr, close + nloss)
    elif close > prev_atr:
        return close - nloss
    else:
        return close + nloss


def calculateEMA(src, length):
    alpha = 2 / (length + 1)

    # Initialize sum with the Simple Moving Average (SMA) for the first value
    sma_first_value = src.head(length).mean()
    sum_value = sma_first_value

    ema_values = []

    for value in src:
        if pd.isna(sum_value):
            sum_value = sma_first_value
        else:
            sum_value = alpha * value + (1 - alpha) * sum_value

        ema_values.append(sum_value)

    return pd.Series(ema_values, index=src.index)


def heikinashi(df: pd.DataFrame) -> pd.DataFrame:
    df_HA = df.copy()
    df_HA['close'] = (df_HA['open'] + df_HA['high'] +
                      df_HA['low'] + df_HA['close']) / 4

    for i in range(0, len(df_HA)):
        if i == 0:
            df_HA.loc[i, 'open'] = (
                (df_HA.loc[i, 'open'] + df_HA.loc[i, 'close']) / 2)
        else:
            df_HA.loc[i, 'open'] = (
                (df_HA.loc[i-1, 'open'] + df_HA.loc[i-1, 'close']) / 2)

    df_HA['high'] = df_HA[['open', 'close', 'high']].max(axis=1)
    df_HA['low'] = df_HA[['open', 'close', 'low']].min(axis=1)

    return df_HA


def calculate_ut_bot(df, SENSITIVITY, ATR_PERIOD):
    dataframe = df.copy()
    # UT Bot Parameters
    SENSITIVITY = SENSITIVITY
    ATR_PERIOD = ATR_PERIOD

    # Compute ATR And nLoss variable
    dataframe["xATR"] = ta.ATR(
        dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=ATR_PERIOD)
    dataframe["nLoss"] = SENSITIVITY * dataframe["xATR"]

    dataframe["ATRTrailingStop"] = [0.0] + \
        [np.nan for i in range(len(dataframe) - 1)]

    for i in range(1, len(dataframe)):
        dataframe.loc[i, "ATRTrailingStop"] = xATRTrailingStop_func(
            dataframe.loc[i, "close"],
            dataframe.loc[i - 1, "close"],
            dataframe.loc[i - 1, "ATRTrailingStop"],
            dataframe.loc[i, "nLoss"],
        )

    dataframe['Ema'] = calculateEMA(dataframe['close'], 1)
    dataframe["Above"] = calculate_crossover(
        dataframe['Ema'], dataframe["ATRTrailingStop"])
    dataframe["Below"] = calculate_crossover(
        dataframe["ATRTrailingStop"], dataframe['Ema'])

    # Buy Signal
    dataframe.loc[
        (
            (dataframe["close"] > dataframe["ATRTrailingStop"])
            &
            (dataframe["Above"] == True)
        ),
        'UT_Signal'] = 1

    dataframe.loc[
        (
            (dataframe["close"] < dataframe["ATRTrailingStop"])
            &
            (dataframe["Below"] == True)
        ),
        'UT_Signal'] = -1

    return dataframe['UT_Signal']


def calculate_crossover(source1, source2):
    return (source1 > source2) & (source1.shift(1) <= source2.shift(1))
