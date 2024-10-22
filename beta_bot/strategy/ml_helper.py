import numpy as np
import pandas as pd

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import math
from enum import Enum
from beta_bot.strategy.adaptive_trend import adaptiveTrendFinder
from beta_bot.strategy.stc_indicator import calculateSTCIndicator
from beta_bot.strategy.ut_signal import calculate_ut_bot
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
from beta_bot.temp_storage import temp_storage_data, TempStorage


def calculate_cci_improved(
    source1: pd.Series, source2: pd.Series, source3: pd.Series, length
) -> pd.Series:
    source1: np.ndarray = source1.to_numpy()
    source2: np.ndarray = source2.to_numpy()
    source3: np.ndarray = source3.to_numpy()

    windows = np.lib.stride_tricks.sliding_window_view(source1, window_shape=(length,))
    source3_modify = source3[:, np.newaxis][length - 1 :]
    sums = np.sum(np.abs(windows - source3_modify), axis=1)

    mad = sums / length

    mad_series = pd.Series(index=range(len(source1)), dtype=float)
    mad_series[-len(mad) :] = mad
    mad_series = mad_series.to_numpy()
    mcci = source2 / mad_series / 0.015
    return mcci


def rescale(src, old_min, old_max, new_min, new_max):
    return new_min + (new_max - new_min) * (src - old_min) / np.maximum(
        (old_max - old_min), 10e-10
    )


def normalize_optimized(min_val, max_val, source: pd.Series):
    source = source.to_numpy()
    historic_min = 10e10
    historic_max = -10e10
    # source.fillna(historic_min)
    src_filled_min = np.nan_to_num(source, historic_min)
    # source.fillna(historic_max)
    src_filled_max = np.nan_to_num(source, historic_max)
    historic_min = np.minimum.accumulate(src_filled_min)
    historic_max = np.maximum.accumulate(src_filled_max)
    division_value = np.maximum((historic_max - historic_min), 10e-10)
    normalized_src = (
        min_val + (max_val - min_val) * (source - historic_min)
    ) / division_value
    return normalized_src


def n_rsi(src, n1, n2):
    rsi = ta.RSI(src, n1)
    ema_rsi = ta.EMA(rsi, n2)
    return rescale(ema_rsi, 0, 100, 0, 1)


def n_cci(dataframe, n1, n2):
    df = dataframe.copy()
    source = df["close"]

    df["mas"] = ta.SMA(source, n1)
    df["diffs"] = source - df["mas"]
    df["cci"] = pd.Series(
        calculate_cci_improved(dataframe["open"], df["diffs"], df["mas"], n1)
    )

    df["ema_cci"] = ta.EMA(df["cci"], n2)

    normalized_wt_diff = pd.Series(normalize_optimized(0, 1, df["ema_cci"]))
    return normalized_wt_diff


def n_wt(src, n1, n2):
    ema1 = ta.EMA(src, n1)
    ema2 = ta.EMA(np.abs(src - ema1), n1)
    ci = (src - ema1) / (0.015 * ema2)
    wt1 = ta.EMA(ci, n2)
    wt2 = ta.SMA(wt1, 4)
    diff = wt1 - wt2
    normalized_wt_diff = pd.Series(normalize_optimized(0, 1, pd.Series(diff)))
    return normalized_wt_diff


def calculate_tr_optimized(high, low, close):
    previos_close = np.roll(close, 1)

    diff_h_n_l = high - low
    abs_value_h_n_c = np.abs(high - previos_close)
    abs_value_h_n_c[0] = abs(high[0] - 0)
    abs_value_l_n_c = np.abs(low - previos_close)
    abs_value_l_n_c[0] = abs(low[0] - 0)
    tr = np.maximum(np.maximum(diff_h_n_l, abs_value_h_n_c), abs_value_l_n_c)
    return tr


def calculate_directionalMovementPlus_optimized(high, low):
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)

    diff_h_n_ph = high - prev_high
    diff_h_n_ph[0] = high[0] - 0
    diff_pl_n_l = prev_low - low
    diff_pl_n_l[0] = 0 - low[0]
    dmp_value = np.maximum(diff_h_n_ph, 0) * (diff_h_n_ph > diff_pl_n_l)
    return dmp_value


def calculate_negMovement_optimized(high, low):
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)

    diff_h_n_ph = high - prev_high
    diff_h_n_ph[0] = high[0] - 0
    diff_pl_n_l = prev_low - low
    diff_pl_n_l[0] = 0 - low[0]
    negMovement = np.maximum(diff_pl_n_l, 0) * (diff_pl_n_l > diff_h_n_ph)
    return negMovement


def n_adx_optimized(
    highSrc: pd.Series, lowSrc: pd.Series, closeSrc: pd.Series, n1: int
):
    length = n1
    highSrc_numpy = highSrc.to_numpy()
    lowSrc_numpy = lowSrc.to_numpy()
    closeSrc_numpy = closeSrc.to_numpy()

    tr = calculate_tr_optimized(highSrc_numpy, lowSrc_numpy, closeSrc_numpy)
    directionalMovementPlus = calculate_directionalMovementPlus_optimized(
        highSrc_numpy, lowSrc_numpy
    )
    negMovement = calculate_negMovement_optimized(highSrc_numpy, lowSrc_numpy)

    trSmooth = np.zeros_like(closeSrc_numpy)
    trSmooth[0] = np.nan
    for i in range(0, len(tr)):
        trSmooth[i] = trSmooth[i - 1] - trSmooth[i - 1] / length + tr[i]

    smoothDirectionalMovementPlus = np.zeros_like(closeSrc)
    smoothDirectionalMovementPlus[0] = np.nan
    for i in range(0, len(directionalMovementPlus)):
        smoothDirectionalMovementPlus[i] = (
            smoothDirectionalMovementPlus[i - 1]
            - smoothDirectionalMovementPlus[i - 1] / length
            + directionalMovementPlus[i]
        )

    smoothnegMovement = np.zeros_like(closeSrc)
    smoothnegMovement[0] = np.nan
    for i in range(0, len(negMovement)):
        smoothnegMovement[i] = (
            smoothnegMovement[i - 1]
            - smoothnegMovement[i - 1] / length
            + negMovement[i]
        )

    diPositive = smoothDirectionalMovementPlus / trSmooth * 100
    diNegative = smoothnegMovement / trSmooth * 100
    dx = np.abs(diPositive - diNegative) / (diPositive + diNegative) * 100
    dx_series = pd.Series(dx)

    adx = dx_series.copy()
    adx.iloc[:length] = adx.rolling(length).mean().iloc[:length]
    adx = adx.ewm(alpha=(1.0 / length), adjust=False).mean()
    return rescale(adx, 0, 100, 0, 1)


def heikinashi(df: pd.DataFrame) -> pd.DataFrame:
    df_HA = df.copy()
    df_HA["close"] = (df_HA["open"] + df_HA["high"] + df_HA["low"] + df_HA["close"]) / 4

    for i in range(0, len(df_HA)):
        if i == 0:
            df_HA.loc[i, "open"] = (df_HA.loc[i, "open"] + df_HA.loc[i, "close"]) / 2
        else:
            df_HA.loc[i, "open"] = (
                df_HA.loc[i - 1, "open"] + df_HA.loc[i - 1, "close"]
            ) / 2

    df_HA["high"] = df_HA[["open", "close", "high"]].max(axis=1)
    df_HA["low"] = df_HA[["open", "close", "low"]].min(axis=1)

    return df_HA


class FeatureName(Enum):
    rsi = "RSI"
    wt = "WT"
    cci = "CCI"
    adx = "ADX"
    ema = "EMA"
    sma = "SMA"
    macd = "MACD"
    tema = "TEMA"
    open = "OPEN"
    high = "HIGH"
    low = "LOW"
    close = "CLOSE"
    volume = "VOLUME"
    all = "ALL"


def chooseFeatureName(name: FeatureName, dataframe, paramsA, paramsB):
    df = dataframe.copy()
    source = df["open"]
    hlc3 = (df["high"] + df["low"] + df["open"]) / 3

    if name == FeatureName.open.name:
        return df["open"]
    if name == FeatureName.high.name:
        return df["high"]
    if name == FeatureName.low.name:
        return df["low"]
    if name == FeatureName.close.name:
        return df["close"]
    if name == FeatureName.volume.name:
        return df["volume"]
    if name == FeatureName.rsi.name:
        return n_rsi(source, paramsA, paramsB)
    if name == FeatureName.wt.name:
        return n_wt(hlc3, paramsA, paramsB)
    if name == FeatureName.cci.name:
        return n_cci(df, paramsA, paramsB)
    if name == FeatureName.adx.name:
        return n_adx_optimized(df["high"], df["low"], df["open"], paramsA)
    if name == FeatureName.ema.name:
        ema = ta.EMA(df, paramsA) - ta.EMA(df, paramsB)
        old_min = ema.min()
        old_max = ema.max()
        return rescale(ema, old_min, old_max, 0, 1)
    if name == FeatureName.sma.name:
        sma = ta.SMA(df, paramsA) - ta.SMA(df, paramsB)
        old_min = sma.min()
        old_max = sma.max()
        return rescale(sma, old_min, old_max, 0, 1)
    if name == FeatureName.macd.name:
        macd = ta.MACD(df)["macdhist"]
        old_min = macd.min()
        old_max = macd.max()
        return rescale(macd, old_min, old_max, 0, 1)
    if name == FeatureName.tema.name:
        tema = ta.TEMA(df, paramsA) - ta.TEMA(df, paramsB)
        old_min = tema.min()
        old_max = tema.max()
        return rescale(tema, old_min, old_max, 0, 1)
    if name == FeatureName.all.name:
        all_value = (
            n_rsi(source, paramsA, paramsB)
            + n_wt(hlc3, paramsA, paramsB)
            + n_cci(df, paramsA, paramsB)
            + n_adx_optimized(df["high"], df["low"], df["open"], paramsA)
        )
        old_min = all_value.min()
        old_max = all_value.max()
        return rescale(all_value, old_min, old_max, 0, 1)


def extract_features(dataframe: pd.DataFrame, training_params):
    df = dataframe.copy()
    future_count = training_params["future_count"]
    feature_count = training_params["feature_count"]
    df_shifted = df.shift(future_count)

    for i in range(1, feature_count + 1):
        df[f"f{i}"] = chooseFeatureName(
            training_params[f"f{i}"]["name"],
            df,
            training_params[f"f{i}"]["paramsA"],
            training_params[f"f{i}"]["paramsB"],
        )
    df["y_train"] = (df_shifted["close"] < dataframe["close"]).astype(int)
    df["y_train"] = df["y_train"].shift(-future_count)
    df.loc[len(df) - (future_count) :, ["y_train"]] = np.nan
    return df


def filter_volatility(dataframe: pd.DataFrame, minLength: int, maxLength: int):
    df = dataframe.copy()
    recentAtr = ta.ATR(df["high"], df["low"], df["close"], timeperiod=minLength)
    historicalAtr = ta.ATR(df["high"], df["low"], df["close"], timeperiod=maxLength)
    return recentAtr > historicalAtr


def regime_filter(dataframe, threshold):
    df = dataframe.copy()
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    src = ohlc4

    value1 = pd.Series(0, index=df["close"].index, dtype=float)
    value2 = pd.Series(0, index=df["close"].index, dtype=float)
    klmf = pd.Series(0, index=df["close"].index, dtype=float)

    for i in range(0, len(value1)):
        if i == 0:
            value1[i] = 0
        else:
            value1[i] = 0.2 * (src[i] - src[i - 1]) + 0.8 * value1[i - 1]

    for i in range(0, len(value1)):
        if i == 0:
            value2[i] = 0.1 * (df["high"][i] - df["low"][i]) + 0.8 * 0
        else:
            value2[i] = 0.1 * (df["high"][i] - df["low"][i]) + 0.8 * value2[i - 1]

    omega = abs(value1 / value2)
    alpha = (-(omega**2) + np.sqrt(omega**4 + 16 * omega**2)) / 8

    for i in range(0, len(value1)):
        if i == 0:
            klmf[i] = alpha[i] * src[i] + (1 - alpha[i]) * 0
        else:
            klmf[i] = alpha[i] * src[i] + (1 - alpha[i]) * klmf[i - 1]

    absCurveSlope = klmf.diff().abs()
    exponentialAverageAbsCurveSlope = 1.0 * ta.EMA(absCurveSlope, 200)
    normalized_slope_decline = (
        absCurveSlope - exponentialAverageAbsCurveSlope
    ) / exponentialAverageAbsCurveSlope
    return normalized_slope_decline >= threshold


def ema_filter(dataframe, period):
    df = dataframe.copy()
    ema = ta.EMA(df["close"], period)
    filter_value = (df["close"] > ema).astype(int) - (df["close"] < ema).astype(int)
    return filter_value


def sma_filter(dataframe, period):
    df = dataframe.copy()
    sma = ta.SMA(df["close"], period)
    filter_value = (df["close"] > sma).astype(int) - (df["close"] < sma).astype(int)
    return filter_value


def kernel_filter(dataframe, loopback, relative_weight, start_at_bar):
    df = dataframe.copy()
    khat1 = pd.Series(
        rational_quadratic(df["close"], loopback, relative_weight, start_at_bar)
    )
    # wasBearishRate = khat1.shift(2) > khat1.shift(1)
    # isBearishRate = khat1.shift(1) > khat1
    # wasBullishRate = khat1.shift(2) < khat1.shift(1)
    filter_rate = (khat1.shift(1) < khat1).astype(int) - (
        khat1.shift(1) > khat1
    ).astype(int)
    return filter_rate


def rational_quadratic(
    price_feed: np.ndarray,
    lookback: int,
    relative_weight: float,
    start_at_bar: int,
) -> np.ndarray:
    length_of_prices = len(price_feed)
    bars_calculated = start_at_bar + 1

    result = np.zeros(length_of_prices, dtype=float)
    lookback_squared = np.power(lookback, 2)
    denominator = lookback_squared * 2 * relative_weight

    for index in range(length_of_prices):
        current_weight = 0.0
        cumulative_weight = 0.0

        for i in range(bars_calculated):
            y = np.nan if (index - i) < 0 else price_feed[index - i]
            w = np.power(
                1 + (np.power(i, 2) / denominator),
                -relative_weight,
            )
            current_weight += y * w
            cumulative_weight += w

        result[index] = current_weight / cumulative_weight

    return result


def gaussian(
    price_feed: np.ndarray,
    lookback: int,
    start_at_bar: int,
) -> np.ndarray:
    length_of_prices = len(price_feed)
    bars_calculated = start_at_bar + 1

    result = np.zeros(length_of_prices, dtype=float)
    lookback_squared = np.power(lookback, 2)
    denominator = lookback_squared * 2

    for index in range(length_of_prices):
        current_weight = 0.0
        cumulative_weight = 0.0

        for i in range(bars_calculated):
            y = np.nan if (index - i) < 0 else price_feed[index - i]
            w = np.exp(-(np.power(i, 2) / denominator))
            current_weight += y * w
            cumulative_weight += w

        result[index] = current_weight / cumulative_weight

    return result


class Filter(Enum):
    volatility = "filter_volatility"
    regime = "regime_filter"
    trend = "trend_filter"
    stc = "stc_filter"
    ut = "ut_filter"


def getLorentizanDistance(i, current_feature, feature_array):
    feature_distance = math.log(1 + abs(current_feature - feature_array[i]))
    return feature_distance


def fractalFilters(predict_value: pd.Series):
    isDifferentSignalType = predict_value.ne(predict_value.shift())
    return isDifferentSignalType


def compare_value_improved(
    dataframe: pd.DataFrame, dataframe_shifted: pd.DataFrame, future_count: int
):
    df_data = dataframe["close"].to_numpy()
    df_shifted_data = dataframe_shifted["close"].to_numpy()
    result = np.where(df_shifted_data >= df_data, 1, -1)
    result = result[:-future_count]
    result_series = pd.Series(index=range(len(df_data)), dtype=float)
    result_series[: len(result)] = result
    return result_series


def setPredictionAsClearWay(index, dataframe: pd.DataFrame, filter_method):
    df = dataframe.copy()
    global signal_predictions
    prediction_value = 0
    predicted_value = df["predicted_value"].iloc[index]
    filter_value = True

    buy_trend = True
    sell_trend = True

    buy_stc = True
    sell_stc = True

    buy_ut = True
    sell_ut = True

    for filter in filter_method:
        if filter.name == "trend":
            buy_trend = df[filter.value].iloc[index] > 0
            sell_trend = df[filter.value].iloc[index] < 0
        elif filter.name == "stc":
            buy_stc = df[filter.value].iloc[index] > 0
            sell_stc = df[filter.value].iloc[index] < 0
        elif filter.name == "ut":
            buy_ut = df[filter.value].iloc[index] > 0
            sell_ut = df[filter.value].iloc[index] > 0
        else:
            filter_value = (df[filter.value].iloc[index]) and (filter_value)

    if (predicted_value > 0) & filter_value & buy_trend & buy_stc & buy_ut:
        prediction_value = 1
    elif (predicted_value < 0) & filter_value & sell_trend & sell_stc & sell_ut:
        prediction_value = -1
    else:
        if index == 0:
            prediction_value = 0
        else:
            prediction_value = signal_predictions[index - 1]
    signal_predictions[index] = prediction_value
    return prediction_value


def train_model(dataframe, training_params):
    # Variable To Use
    feature_count = training_params["feature_count"]
    future_count = training_params["future_count"]
    num_candles_to_fetch = temp_storage_data[TempStorage.config]["exchange"][
        "ohlcv_candle_limit"
    ]

    # Copy Dataframe
    df = dataframe.copy()
    df_for_predict = dataframe.copy()

    # Dropping NAN Value
    df.dropna(inplace=True)
    df_for_predict.fillna(0, inplace=True)

    # Extract Feature List In data_x and data_y for training and predicting
    feature_list = [f"f{i}" for i in range(1, feature_count + 1)]
    data_x = df[feature_list].values
    data_y = df["y_train"].values
    data_for_predict = df_for_predict[feature_list].values

    # Scaling Using StandardScaler
    scaler_for_train = StandardScaler()
    data_x = scaler_for_train.fit_transform(data_x)

    scaler_for_test = StandardScaler()
    data_for_predict = scaler_for_test.fit_transform(data_for_predict)

    # Split Data For Training
    X_train, X_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=60 / num_candles_to_fetch, random_state=42
    )
    # Build Model
    xgb_model = XGBClassifier(
        tree_method="auto",
    )
    model = xgb_model
    a = datetime.now()
    model.fit(data_x, data_y)
    joblib.dump(model, "../../trained_model.pkl")
    b = datetime.now()
    # Predict Model
    length_to_predict = data_for_predict.shape[0] - future_count
    current_predict_class = model.predict(data_for_predict[:length_to_predict])
    current_predict_class[current_predict_class == 0] = -1
    current_predict_probability = model.predict_proba(
        data_for_predict[:length_to_predict]
    )
    for _ in range(length_to_predict, length_to_predict + future_count):
        current_predict_class = np.append(current_predict_class, 0)
        current_predict_probability = np.vstack([current_predict_probability, [0, 0]])
    current_predict_probability = np.amax(
        current_predict_probability, axis=1, keepdims=True
    )

    # Testing Model Performance
    y_pred = model.predict(X_test)
    print(f"Accuracy Test : {accuracy_score(y_test,y_pred)}")
    print(f"speed code is {round((b-a).microseconds * 0.001, 4)}ms")
    return current_predict_class, current_predict_probability


def predict_future(dataframe: pd.DataFrame, training_params):
    df = dataframe.copy()
    global signal_predictions
    signal_predictions = {}

    df["predicted_value"], df["predicted_proba"] = train_model(df, training_params)
    df["isDifferentSignalType"] = fractalFilters(df["predicted_value"])
    dataframe["predicted_value"] = df["predicted_value"]
    dataframe["predicted_proba"] = df["predicted_proba"]
    dataframe["buy_signal"] = (df["predicted_value"] > 0) & (
        df["isDifferentSignalType"]
    )
    dataframe["sell_signal"] = (df["predicted_value"] < 0) & (
        df["isDifferentSignalType"]
    )
    return dataframe


def mlRunModel(dataframe, training_params):
    df = dataframe.copy()
    df = extract_features(df, training_params)
    df = predict_future(df, training_params)
    return df


distances = []
predictions = []
signal_predictions = {}
