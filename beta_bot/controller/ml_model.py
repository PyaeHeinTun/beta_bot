import pandas as pd
import talib as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from beta_bot.controller import base as controller
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from beta_bot.strategy import base as strategy
from beta_bot.temp_storage import temp_storage_data, TempStorage


def split_predictions_with_proba(
    current_predict_class, current_predict_probability, dataframe, n_predict_candle
) -> dict:
    final_predictions = {}
    df_for_predict = dataframe.copy()

    current_predictions_with_prob = zip(
        current_predict_class, current_predict_probability
    )

    for i, (pred_class, prob) in enumerate(current_predictions_with_prob, 1):
        predict_minute = (
            df_for_predict.iloc[i - 1]["date"].minute + n_predict_candle
        ) % 60
        predict_class = pred_class
        predict_probability = prob

        final_predictions[predict_minute] = {
            "index": i,
            "class": predict_class,
            "probability": predict_probability,
        }
    return final_predictions


def predict_future(dataframe: pd.DataFrame, symbol) -> dict:
    df = dataframe.copy()
    df.drop(df.tail(1).index, inplace=True)
    training_params = temp_storage_data[TempStorage.config]["training_params"]
    complete_df = strategy.mlRunModel(df, training_params)

    last_candle = complete_df.iloc[-2].squeeze()

    return {
        "class": last_candle['predicted_value'],
        "probability": last_candle['predicted_proba'],
    }
