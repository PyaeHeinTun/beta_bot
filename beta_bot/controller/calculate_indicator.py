import pandas as pd
import numpy as np
import math as math
import talib as ta


def calculate_CCI(dataframe, index, diff, ma, p):

    close_array = dataframe['close'].to_numpy()
    close_array = close_array[:index.name]
    diff = diff.to_numpy()
    diff = diff[index.name]
    ma = ma.to_numpy()
    ma = ma[index.name]

    if (len(close_array) < p):
        return 0

    # MAD
    s = 0

    for i in range(len(close_array), len(close_array)-p, -1):
        s = s + abs(dataframe['close'][i] - ma)
    mad = s / p

    if math.isnan(diff):
        return 0
    else:
        return diff
    # Scalping
    mcci = diff/mad/0.015

    return mcci


def minimax(volume, x, p, min, max):
    volume_array = volume.to_numpy()
    volume_array = volume_array[:x.name+1]

    if (len(volume_array) < p):
        return 0

    hi = np.nan_to_num(np.max(volume_array[-p+1:]))
    lo = np.nan_to_num(np.min(volume_array[-p+1:]))

    upper_divisor = (max - min) * (volume_array[len(volume_array)-1] - lo)
    lower_divisor = (hi - lo) + min
    if math.isnan(upper_divisor):
        return 0
    else:
        return upper_divisor
    return upper_divisor/lower_divisor


def scale(mom, index, loopback_period):
    mom_array = mom.to_numpy()
    mom_array = mom_array[:index.name+1]

    if (len(mom_array) < loopback_period):
        return 0

    current_mom = mom[index.name]
    hi = np.nan_to_num(np.max(mom_array[-loopback_period+1:]))
    lo = np.nan_to_num(np.min(mom_array[-loopback_period+1:]))

    return ((current_mom - lo) / (hi - lo)) * 100


def calculate_sm1(s):
    result = []
    for value in s:
        result.append(value if value >= 0 else 0.0)
    return np.sum(result)


def calculate_sm2(s):
    result = []
    for value in s:
        result.append(0.0 if value >= 0 else -value)
    return np.sum(result)


def calculate_mfi_upper(s, x):
    result = np.where(s >= 0, 0.0, x)
    return np.sum(result)


def calculate_mfi_lower(s, x):
    result = np.where(s <= 0, 0.0, x)
    return np.sum(result)


def pine_cmo(src: pd.Series, length):
    # Calculate the momentum (change) of the source data
    mom = src - src.shift(1)

    # Calculate the sum of positive and negative momentum over the specified length
    sm1 = mom.rolling(length).apply(calculate_sm1, raw=True)
    sm2 = mom.rolling(length).apply(calculate_sm2, raw=True)

    # Calculate the Chande Momentum Oscillator (CMO)
    cmo = 100 * ((sm1 - sm2) / (sm1 + sm2))

    return cmo

# =============================================================
# ============== Adaptive Trend Finder ========================
# =============================================================


def adaptiveTrendFinder(dataframe: pd.DataFrame, periods: list[int]):
    df_copy = dataframe.copy()
    df_copy['trend_direction'] = df_copy.apply(
        (lambda x: calculate_trend_direction(x, df_copy, periods)), axis=1)
    df_copy['trend_direction_temp'] = df_copy['trend_direction'].apply(
        lambda x: x[0])

    return df_copy['trend_direction_temp']


def calculate_trend_direction(x, dataframe, periods):
    if (x.name >= periods[2]):
        devMultiplier = 2.0
        # Calculate Deviation,PersionR,Slope,Intercept
        stdDev01, pearsonR01, slope01, intercept01 = calcDev(
            periods[1], dataframe, x.name)
        stdDev02, pearsonR02, slope02, intercept02 = calcDev(
            periods[2], dataframe, x.name)

        # Find the highest Pearson's R
        highestPearsonR = max(pearsonR01, pearsonR02)

        # Determine selected length, slope, intercept, and deviations
        detectedPeriod = 0
        detectedSlope = 0
        detectedIntrcpt = 0
        detectedStdDev = 0

        if highestPearsonR == pearsonR01:
            detectedPeriod = periods[1]
            detectedSlope = slope01
            detectedIntrcpt = intercept01
            detectedStdDev = stdDev01
        elif highestPearsonR == pearsonR02:
            detectedPeriod = periods[2]
            detectedSlope = slope02
            detectedIntrcpt = intercept02
            detectedStdDev = stdDev02
        else:
            # Default case
            raise Exception(f"Cannot Find Highest PearsonR")

        # Calculate start and end price based on detected slope and intercept
        startPrice = math.exp(
            detectedIntrcpt + detectedSlope * (detectedPeriod - 1))
        endPrice = math.exp(detectedIntrcpt)

        trend_direction = endPrice - startPrice
        return (trend_direction, detectedPeriod, highestPearsonR)
    return (0, 0, 0)


def calcDev(length: int, dataframe: pd.DataFrame, index: int):
    logSource = dataframe['close'].apply(lambda x: math.log(x))
    period_1 = length - 1
    sumX = 0.0
    sumXX = 0.0
    sumYX = 0.0
    sumY = 0.0
    for i in range(1, length+1):
        lSrc = logSource[index+1-i]
        sumX += i
        sumXX += i * i
        sumYX += i * lSrc
        sumY += lSrc

    slope = np.nan_to_num((length * sumYX - sumX * sumY) /
                          (length * sumXX - sumX * sumX))
    average = sumY / length
    intercept = average - (slope * sumX / length) + slope
    sumDev = 0.0
    sumDxx = 0.0
    sumDyy = 0.0
    sumDyx = 0.0
    regres = intercept + slope * period_1 * 0.5
    sumSlp = intercept

    for i in range(1, period_1+1):
        lSrc = logSource[index+1-i]
        dxt = lSrc - average
        dyt = sumSlp - regres
        lSrc -= sumSlp
        sumSlp += slope
        sumDxx += dxt * dxt
        sumDyy += dyt * dyt
        sumDyx += dxt * dyt
        sumDev += lSrc * lSrc

    unStdDev = math.sqrt(sumDev / period_1)
    divisor = sumDxx * sumDyy
    if divisor == 0 or np.isnan(divisor):
        pearsonR = 0  # Set Pearson correlation coefficient to NaN
    else:
        pearsonR = sumDyx / math.sqrt(divisor)
    return unStdDev, pearsonR, slope, intercept


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


def calculate_ut_bot(dataframe, SENSITIVITY, ATR_PERIOD):
    # UT Bot Parameters
    SENSITIVITY = SENSITIVITY
    ATR_PERIOD = ATR_PERIOD
    df_copy = dataframe.copy()
    # Compute ATR And nLoss variable
    df_copy["xATR"] = ta.ATR(
        df_copy["high"], df_copy["low"], df_copy["close"], timeperiod=ATR_PERIOD)
    df_copy["nLoss"] = SENSITIVITY * df_copy["xATR"]

    df_copy["ATRTrailingStop"] = [0.0] + \
        [np.nan for i in range(len(df_copy) - 1)]

    for i in range(1, len(df_copy)):
        df_copy.loc[i, "ATRTrailingStop"] = xATRTrailingStop_func(
            df_copy.loc[i, "close"],
            df_copy.loc[i - 1, "close"],
            df_copy.loc[i - 1, "ATRTrailingStop"],
            df_copy.loc[i, "nLoss"],
        )

    df_copy['Ema'] = calculateEMA(df_copy['close'], 1)
    df_copy["Above"] = calculate_crossover(
        df_copy['Ema'], df_copy["ATRTrailingStop"])
    df_copy["Below"] = calculate_crossover(
        df_copy["ATRTrailingStop"], df_copy['Ema'])

    # Buy Signal
    df_copy.loc[
        (
            (df_copy["close"] > df_copy["ATRTrailingStop"])
            &
            (df_copy["Above"] == True)
        ),
        'UT_Signal'] = 1

    df_copy.loc[
        (
            (df_copy["close"] < df_copy["ATRTrailingStop"])
            &
            (df_copy["Below"] == True)
        ),
        'UT_Signal'] = -1

    return df_copy['ATRTrailingStop'], df_copy['Ema']


def calculate_crossover(source1, source2):
    return (source1 > source2) & (source1.shift(1) <= source2.shift(1))

# =============================================================
# ============== Parabolic Sar ================================
# =============================================================


def PSAR(dataframe, af=0.02, max=0.2):
    df_copy = dataframe.copy()
    df_copy.loc[0, 'AF'] = 0.02
    df_copy.loc[0, 'PSAR'] = df_copy.loc[0, 'low']
    df_copy.loc[0, 'EP'] = df_copy.loc[0, 'high']
    df_copy.loc[0, 'PSARdir'] = 1

    for a in range(1, len(df_copy)):
        if df_copy.loc[a-1, 'PSARdir'] == 1:
            df_copy.loc[a, 'PSAR'] = df_copy.loc[a-1, 'PSAR'] + \
                (df_copy.loc[a-1, 'AF'] *
                 (df_copy.loc[a-1, 'EP']-df_copy.loc[a-1, 'PSAR']))
            df_copy.loc[a, 'PSARdir'] = 1

            if df_copy.loc[a, 'low'] < df_copy.loc[a-1, 'PSAR'] or df_copy.loc[a, 'low'] < df_copy.loc[a, 'PSAR']:
                df_copy.loc[a, 'PSARdir'] = -1
                df_copy.loc[a, 'PSAR'] = df_copy.loc[a-1, 'EP']
                df_copy.loc[a, 'EP'] = df_copy.loc[a-1, 'low']
                df_copy.loc[a, 'AF'] = af
            else:
                if df_copy.loc[a, 'high'] > df_copy.loc[a-1, 'EP']:
                    df_copy.loc[a, 'EP'] = df_copy.loc[a, 'high']
                    if df_copy.loc[a-1, 'AF'] <= 0.18:
                        df_copy.loc[a, 'AF'] = df_copy.loc[a-1, 'AF'] + af
                    else:
                        df_copy.loc[a, 'AF'] = df_copy.loc[a-1, 'AF']
                elif df_copy.loc[a, 'high'] <= df_copy.loc[a-1, 'EP']:
                    df_copy.loc[a, 'AF'] = df_copy.loc[a-1, 'AF']
                    df_copy.loc[a, 'EP'] = df_copy.loc[a-1, 'EP']

        elif df_copy.loc[a-1, 'PSARdir'] == -1:
            df_copy.loc[a, 'PSAR'] = df_copy.loc[a-1, 'PSAR'] - \
                (df_copy.loc[a-1, 'AF'] *
                 (df_copy.loc[a-1, 'PSAR']-df_copy.loc[a-1, 'EP']))
            df_copy.loc[a, 'PSARdir'] = -1

            if df_copy.loc[a, 'high'] > df_copy.loc[a-1, 'PSAR'] or df_copy.loc[a, 'high'] > df_copy.loc[a, 'PSAR']:
                df_copy.loc[a, 'PSARdir'] = 1
                df_copy.loc[a, 'PSAR'] = df_copy.loc[a-1, 'EP']
                df_copy.loc[a, 'EP'] = df_copy.loc[a-1, 'high']
                df_copy.loc[a, 'AF'] = af
            else:
                if df_copy.loc[a, 'low'] < df_copy.loc[a-1, 'EP']:
                    df_copy.loc[a, 'EP'] = df_copy.loc[a, 'low']
                    if df_copy.loc[a-1, 'AF'] < max:
                        df_copy.loc[a, 'AF'] = df_copy.loc[a-1, 'AF'] + af
                    else:
                        df_copy.loc[a, 'AF'] = df_copy.loc[a-1, 'AF']

                elif df_copy.loc[a, 'low'] >= df_copy.loc[a-1, 'EP']:
                    df_copy.loc[a, 'AF'] = df_copy.loc[a-1, 'AF']
                    df_copy.loc[a, 'EP'] = df_copy.loc[a-1, 'EP']
    return df_copy['PSAR']


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


def parabolic_sar_ma_strategy(dataframe: pd.DataFrame):
    df_copy = dataframe.copy()
    # Calculate Parabolic SAR
    heikin_ashi = heikinashi(df_copy)
    df_copy['sar'] = PSAR(heikin_ashi)

    return df_copy['sar']

# =============================================================
# ===================== STC Indicator =========================
# =============================================================


def calculateSTCIndicator(dataframe, length, fastLength, slowLength):
    EEEEEE = length
    BBBB = fastLength
    BBBBB = slowLength
    mAAAAA = AAAAA(dataframe, EEEEEE, BBBB, BBBBB)
    return mAAAAA


def AAAA(BBB, BBBB, BBBBB):
    fastMA = ta.EMA(BBB, timeperiod=BBBB)
    slowMA = ta.EMA(BBB, timeperiod=BBBBB)
    AAAA = fastMA - slowMA
    return AAAA


def AAAAA(dataframe, EEEEEE, BBBB, BBBBB):
    AAA = 0.05
    df_copy = dataframe.copy()
    df_copy['DDD'] = 0.0
    df_copy['CCCCC'] = 0
    df_copy['DDDDDD'] = 0
    df_copy['EEEEE'] = 0.0

    df_copy['BBBBBB'] = AAAA(df_copy['close'], BBBB, BBBBB)
    df_copy['CCC'] = df_copy['BBBBBB'].rolling(window=EEEEEE).min()
    df_copy['CCCC'] = df_copy['BBBBBB'].rolling(
        window=EEEEEE).max() - df_copy['CCC']
    df_copy['CCCCC'] = np.where(df_copy['CCCC'] > 0, (df_copy['BBBBBB'] - df_copy['CCC']) /
                                df_copy['CCCC'] * 100, df_copy['CCCCC'].shift(1).fillna(0))

    for i in range(0, len(df_copy)):
        if (i > 0):
            df_copy.at[i, 'DDD'] = df_copy['DDD'].iloc[i-1] + \
                (AAA * (df_copy.at[i, 'CCCCC'] - df_copy['DDD'].iloc[i-1]))

    df_copy['DDDD'] = df_copy['DDD'].rolling(window=EEEEEE).min()
    df_copy['DDDDD'] = df_copy['DDD'].rolling(
        window=EEEEEE).max() - df_copy['DDDD']
    df_copy['DDDDDD'] = np.where(df_copy['DDD'] > 0, (df_copy['DDD'] - df_copy['DDDD']) /
                                 df_copy['DDDDD'] * 100, df_copy['DDD'].fillna(df_copy['DDDDDD'].shift(1)))

    for i in range(0, len(df_copy)):
        if (i > 0):
            df_copy.at[i, 'EEEEE'] = df_copy['EEEEE'].iloc[i-1] + \
                (AAA * (df_copy.at[i, 'DDDDDD'] - df_copy['EEEEE'].iloc[i-1]))

    return df_copy['EEEEE']
