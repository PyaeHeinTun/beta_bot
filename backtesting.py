import asyncio
from beta_bot.rpc import base as rpc
from beta_bot.helper import base as helper
from beta_bot.temp_storage import TempStorage, temp_storage_data
import ccxt.pro as ccxt
import pandas as pd
import mplfinance as mpf
import numpy as np
from beta_bot.controller import base as controller


async def fetch_ohlcv_info(exchange: ccxt.binance, symbol, timeframe, time_range: str) -> pd.DataFrame:
    from_ts = exchange.parse8601(time_range)
    data = await exchange.fetch_ohlcv(
        symbol,
        timeframe,
        since=from_ts,
        params={
            'enableRateLimit': False
        },
    )
    df = pd.DataFrame(
        data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    return df


def trend_signal_and_ha_signal_belowzero(ut: pd.Series, trend: pd.Series, price):
    signal = []
    for index, value in trend.items():
        if (trend[index] < 0):
            print(index, "SELL")
            signal.append(price[index])
        else:
            signal.append(np.nan)
    return signal


def trend_signal_and_ha_signal_abovezero(ut: pd.Series, trend: pd.Series, price):
    signal = []
    for index, value in trend.items():
        if (trend[index] > 0):
            print(index, "BUY")
            signal.append(price[index])
        else:
            signal.append(np.nan)
    return signal


async def run_backtest():
    exchange = ccxt.binance()
    config = temp_storage_data[TempStorage.config]
    exchange.set_sandbox_mode(False)
    exchange.api = config['exchange']['real_key']
    exchange.apiKey = config['exchange']['real_key']
    exchange.secret = config['exchange']['real_secret']
    leverage = config['trade_params']['leverage']
    exchange.options['defaultType'] = config['exchange']['type']
    timeframe = config['timeframe']
    symbols = config['exchange']['pair_whitelist']
    dataframe = await fetch_ohlcv_info(
        exchange, symbols[0], timeframe, '2024-03-06 00:00:00')
    await exchange.close()

    modify_dataframe = dataframe.copy()
    dataframe.index = pd.DatetimeIndex(dataframe['date'])
    ha_dataframe = controller.heikinashi(modify_dataframe)
    modify_dataframe['trend_signal'] = controller.adaptiveTrendFinder(
        ha_dataframe, [0, 2, 2], True)
    modify_dataframe['ut_signal'] = controller.calculate_ut_bot(
        ha_dataframe, 2, 10)

    buy_signal = trend_signal_and_ha_signal_abovezero(
        modify_dataframe['ut_signal'], modify_dataframe['trend_signal'], modify_dataframe['close'])

    sell_signal = trend_signal_and_ha_signal_belowzero(
        modify_dataframe['ut_signal'], modify_dataframe['trend_signal'], modify_dataframe['close'])

    apt = [
        mpf.make_addplot(
            buy_signal, type='scatter', markersize=200, marker='^'),
        mpf.make_addplot(
            sell_signal, type='scatter', markersize=200, marker='v')
    ]

    mpf.plot(dataframe, type="candle", addplot=apt)


async def main():
    config = helper.read_config()
    temp_storage_data[TempStorage.config] = config

    bot_task = asyncio.create_task(run_backtest())

    done, pending = await asyncio.wait([bot_task], return_when=asyncio.FIRST_COMPLETED)

    for task in pending:
        task.cancel()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
