import pandas as pd
from beta_bot.model.base import Trade
from datetime import datetime
from beta_bot.database import base as database
from beta_bot.helper import base as helper
import sqlite3
import ccxt
from beta_bot.temp_storage import TempStorage, temp_storage_data
from beta_bot.rpc import base as rpc


async def main_trade_func(dataframe: pd.DataFrame, future_predictions: dict, pair: str) -> Trade:
    cursor: sqlite3.Cursor = temp_storage_data[TempStorage.cursor]
    conn: sqlite3.Connection = temp_storage_data[TempStorage.conn]

    leverage = temp_storage_data[TempStorage.config]['trade_params']['leverage']
    stake_ammount = temp_storage_data[TempStorage.config]['trade_params']['stake_ammount']
    current_price = dataframe.iloc[-1].squeeze()['close']
    trades_list = _find_existing_trade(cursor, conn)
    previous_candle_1 = dataframe.iloc[-2].squeeze()

    if (len(trades_list) != 0):

        for trade in trades_list:
            _update_current_price_in_trade(cursor, conn, current_price,trade)
            if (_custom_exit(dataframe, trade) & (trade.pair == pair)):
                await _close_position(cursor, conn, trade)

        return trades_list

    else:
        signal, should_trade = _populate_trade_entry(
            dataframe, future_predictions,pair)
        if (should_trade):
            loopback_1_candle_min = previous_candle_1['low']
            loopback_1_candle_max = previous_candle_1['high']
            new_trade_short = _get_trade_object(
                1, current_price, leverage, pair, stake_ammount, loopback_1_candle_min, loopback_1_candle_max, signal)
            new_trade_long = _get_trade_object(
                0, current_price, leverage, pair, stake_ammount, loopback_1_candle_min, loopback_1_candle_max, signal)
            
            if (signal == "long"):
                await _create_position(cursor, conn, new_trade_long)
            if (signal == "short"):
                await _create_position(cursor, conn, new_trade_short)

            config = helper.read_config()
            telegramEnabled = config["telegram"]["enabled"]
            if (telegramEnabled):
                await rpc.send_message_for_trade(temp_storage_data[TempStorage.telegramBot])
            
        return trades_list


def _custom_exit(dataframe: pd.DataFrame, trade: Trade) -> bool:
    exit_reason = "none"
    deviation_for_hedging = 0
    order_book = temp_storage_data[f'order_book{trade.pair}']
    current_profit = trade.calculate_profit_ratio()
    current_profit_roi = current_profit['roi']
    current_price = dataframe.iloc[-1].squeeze()['close']
    trade_open_rate = trade.entry_price

    previous_candle_max_value = trade.p_max
    previous_candle_min_value = trade.p_min
    deviation_open_n_high = helper.calculate_deviation(
        trade.entry_price, previous_candle_max_value, order_book)
    deviation_open_n_low = helper.calculate_deviation(
        trade.entry_price, previous_candle_min_value, order_book)

    if (trade.signal == "long"):
        deviation_for_hedging = deviation_open_n_low
    else:
        deviation_for_hedging = deviation_open_n_high

    price_to_exit_profit_long = 0
    price_to_exit_loss_long = 0
    price_to_exit_profit_short = 0
    price_to_exit_loss_short = 0
    if (trade.is_short == 0):
        price_to_exit_profit_long = helper.add_deviation(order_book,
                                                         trade_open_rate, False, int((deviation_for_hedging)*0.5))
        price_to_exit_loss_long = helper.add_deviation(order_book,
                                                       trade_open_rate, True, int((deviation_for_hedging)*0.5))
    else:
        price_to_exit_profit_short = helper.add_deviation(order_book,
                                                          trade_open_rate, True, int((deviation_for_hedging)*0.5))
        price_to_exit_loss_short = helper.add_deviation(order_book,
                                                        trade_open_rate, False, int((deviation_for_hedging)*0.5))

    if (current_profit_roi > 0):
        if ((trade.is_short == 0) & (current_price > price_to_exit_profit_long)):
            exit_reason = "swp"
            return _confirm_trade_exit(exit_reason)
        if ((trade.is_short == 1) & (current_price < price_to_exit_profit_short)):
            exit_reason = "swp"
            return _confirm_trade_exit(exit_reason)

    if (current_profit_roi < 0):
        if ((trade.is_short == 0) & (current_price < price_to_exit_loss_long)):
            exit_reason = "swl"
            return _confirm_trade_exit(exit_reason)
        if ((trade.is_short == 1) & (current_price > price_to_exit_loss_short)):
            exit_reason = "swl"
            return _confirm_trade_exit(exit_reason)

    return _confirm_trade_exit("none")


def _populate_trade_entry(dataframe: pd.DataFrame, future_predictions: dict, symbol:str) -> bool:
    cursor: sqlite3.Cursor = temp_storage_data[TempStorage.cursor]
    conn: sqlite3.Connection = temp_storage_data[TempStorage.conn]
    trades_list = _get_last_trade(cursor, conn)

    if (len(trades_list) != 0):
        if (datetime.strptime(trades_list[0].updated_at, "%Y-%m-%d %H:%M:%S.%f").minute == datetime.utcnow().minute):
            return ("none", False)

    signal = "none"
    predictions = 0
    probability = 0

    predictions = future_predictions[symbol]['class']
    probability = future_predictions[symbol]['probability']

    if (predictions == 1 and probability > 0.5):
        signal = "long"
    elif (predictions == -1 and probability > 0.5):
        signal = "short"
    else:
        signal = "none"

    is_best_time = helper.is_quarter_hour(datetime.utcnow())

    if ((signal == "long") & is_best_time):
        return ("long", True)
    elif ((signal == "short") & is_best_time):
        return ("short", True)
    else:
        return ("none", False)


def _confirm_trade_exit(exit_reason: str) -> bool:
    if (exit_reason == 'swp'):
        return True
    if (exit_reason == 'swl'):
        return True
    else:
        return False


async def _create_position(cursor: sqlite3.Cursor, conn: sqlite3.Connection, trade: Trade):
    is_dry_run = temp_storage_data[TempStorage.config]['dry_run']
    exchange: ccxt.binance = temp_storage_data[TempStorage.exchange]
    order_book = temp_storage_data[f'order_book{trade.pair}']
    quantity = (trade.stake_ammount * trade.leverage) / trade.entry_price

    if (trade.is_short == 1):
        order_short = await exchange.create_order(
            symbol=trade.pair,
            side="sell",
            type="market",
            amount=quantity,
            # price=order_book['bids'][0][0],
            params={
                # 'hedged': 'true',
                'positionSide': 'SHORT',
                'leverage': trade.leverage
            })
        database.create_trade(cursor, conn, order_short.get('price'), trade.exit_price,
                                1, trade.created_at, trade.updated_at, trade.leverage, trade.stake_ammount, trade.pair, trade.is_completed, trade.p_min, trade.p_max, trade.signal)
    elif (trade.is_short == 0):
        order_long = await exchange.create_order(
            symbol=trade.pair,
            side="buy",
            type="market",
            amount=quantity,
            # price=order_book['asks'][0][0],
            params={
                # 'hedged': 'true',
                'positionSide': 'LONG',
                'leverage': trade.leverage
            })
        database.create_trade(cursor, conn, order_long.get('price'), trade.exit_price,
                                0, trade.created_at, trade.updated_at, trade.leverage, trade.stake_ammount, trade.pair, trade.is_completed, trade.p_min, trade.p_max, trade.signal)
    return trade


async def _close_position(cursor: sqlite3.Cursor, conn: sqlite3.Connection, trade: Trade):
    is_dry_run = temp_storage_data[TempStorage.config]['dry_run']
    exchange: ccxt.binance = temp_storage_data[TempStorage.exchange]
    quantity = (trade.stake_ammount * trade.leverage) / trade.entry_price

    if (trade.is_short):
        await exchange.create_order(
            symbol=trade.pair,
            side="buy",
            type="market",
            amount=quantity,
            params={
                # 'hedged': 'true',
                'positionSide': 'SHORT',
                'leverage': trade.leverage,
            })
    else:
        await exchange.create_order(
            symbol=trade.pair,
            side="sell",
            type="market",
            amount=quantity,
            params={
                # 'hedged': 'true',
                'positionSide': 'LONG',
                'leverage': trade.leverage,
            })

    database.update_pending_to_completed_trade(cursor, conn, trade.id)
    config = helper.read_config()
    telegramEnabled = config["telegram"]["enabled"]
    if (telegramEnabled):
        await rpc.send_message_for_exit(temp_storage_data[TempStorage.telegramBot])
    return trade


def _update_current_price_in_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection, current_price: float,trade:Trade):
    database.update_trade_current_price(cursor, conn, current_price,trade)
    return


def _find_existing_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    trades_list = database.find_current_trade(cursor, conn)

    trades_list = [database.map_tuple_into_trade(
        value[0]) for value in zip(trades_list)]

    return trades_list


def _get_last_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    trades_list = database.find_last_trade(cursor, conn)

    trades_list = [database.map_tuple_into_trade(
        value[0]) for value in zip(trades_list)]

    return trades_list


def _get_trade_object(is_short: int, current_price, leverage, pair, stake_ammount, p_min, p_max, signal) -> Trade:
    return Trade(
        trade_id=0,
        updated_at=datetime.utcnow(),
        created_at=datetime.utcnow(),
        entry_price=current_price,
        exit_price=current_price,
        is_short=is_short,
        leverage=leverage,
        pair=pair,
        stake_ammount=stake_ammount,
        is_completed=0,
        p_min=p_min,
        p_max=p_max,
        signal=signal
    )
