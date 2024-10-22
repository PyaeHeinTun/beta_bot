import sqlite3
from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.utils.markdown import hbold
from aiogram.types import Message
from beta_bot.temp_storage import temp_storage_data, TempStorage
from datetime import datetime, timedelta
from beta_bot.database import base as database
import ccxt
from prettytable import PrettyTable
from beta_bot.helper import base as helper
from beta_bot.rpc import base as rpc


def _find_completed_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    trades_list = database.find_completed_trade(cursor, conn)

    trades_list = [database.map_tuple_into_trade(
        value[0]) for value in zip(trades_list)]

    return trades_list


def _find_completed_last_seven_days(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    seven_days_ago = datetime.utcnow() - timedelta(days=7)

    # Construct the SQL query to fetch data from the last 7 days
    query = "SELECT * FROM trades WHERE updated_at >= ?"
    params = (seven_days_ago,)

    # Execute the query
    cursor.execute(query, params)
    trades_list = cursor.fetchall()
    trades_list = [database.map_tuple_into_trade(
        value[0]) for value in zip(trades_list)]
    return trades_list


def _get_last_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    trades_list = database.find_last_trade(cursor, conn)

    trades_list = [database.map_tuple_into_trade(
        value[0]) for value in zip(trades_list)]

    return trades_list


def command_profit(message: Message):
    _cursor = temp_storage_data[TempStorage.cursor]
    _conn = temp_storage_data[TempStorage.conn]
    bot_start_date = temp_storage_data[TempStorage.botStartDate]

    trades_list = _find_completed_trade(_cursor, _conn)
    lastest_trade = _get_last_trade(_cursor, _conn)

    profit_closed_pnl = round(sum(item.calculate_profit_ratio()[
        'pnl'] for item in trades_list), 2)
    profit_closed_roi = round(sum(item.calculate_profit_ratio()[
        'roi'] for item in trades_list), 2)
    trade_count = len(trades_list)

    if len(trades_list) == 0:
        markdown_msg = f"No trades yet.\n*Bot started:* `{bot_start_date}`"
    else:
        latest_trade_date = f"({lastest_trade[0].created_at})"
        all_win_trade = sum(
            1 for item in trades_list if item.calculate_profit_ratio()['roi'] > 0)
        all_lose_trade = sum(
            1 for item in trades_list if item.calculate_profit_ratio()['roi'] < 0)
        winrate = all_win_trade / trade_count
        markdown_msg = (
            f"*Total Trade Count:* `{trade_count}`\n"
            f"*Total PNL* `{profit_closed_pnl}USDT`\n"
            f"*Total ROI* `{profit_closed_roi}%`\n"
            f"*Bot started:* `{bot_start_date}`\n"
            f"*Latest Trade opened:* `{latest_trade_date}`\n"
            f"*Win / Loss:* `{all_win_trade} / {all_lose_trade}`\n"
            f"*Winrate:* `{winrate:.2%}`\n"
        )
    return markdown_msg, ParseMode.MARKDOWN


def command_status():
    _cursor = temp_storage_data[TempStorage.cursor]
    _conn = temp_storage_data[TempStorage.conn]
    trades_list = database.find_current_trade(_cursor, _conn)

    trades_list = [database.map_tuple_into_trade(
        value[0]) for value in zip(trades_list)]

    table = PrettyTable()
    table.field_names = ["id", "pair", "pnl", "mode"]

    if (len(trades_list) == 0):
        table.add_row(["0", "Searching", "0.0", "N"])
    else:
        for trade in trades_list:
            profit = trade.calculate_profit_ratio()
            id = str(trade.id)
            pair = trade.pair.split("/")[0]
            mode = "S" if trade.is_short == 1 else "L"
            pnl = profit['pnl']
            table.add_row([id, pair, pnl, mode])

    table.align = "r"
    table_string = '```\n{}```'.format(table.get_string())
    message_output = table_string
    return message_output, ParseMode.MARKDOWN


async def send_message_for_trade(bot: Bot):
    config = temp_storage_data[TempStorage.config]
    chat_id = config['telegram']['chat_id']
    # output_message, parse_mode = rpc.command_status()
    await bot.send_message(chat_id=chat_id, text="You have a new trade",
                           parse_mode=ParseMode.HTML)


async def send_message_for_exit(bot: Bot):
    config = temp_storage_data[TempStorage.config]
    chat_id = config['telegram']['chat_id']
    # output_message, parse_mode = rpc.command_status()
    await bot.send_message(chat_id=chat_id, text="You have an existed trade",
                           parse_mode=ParseMode.HTML)


def command_daily(message: Message):
    _cursor = temp_storage_data[TempStorage.cursor]
    _conn = temp_storage_data[TempStorage.conn]
    daily_profit = {}
    daily_total_trades = {}
    daily_total_win = {}
    daily_total_lose = {}

    trades_list = _find_completed_last_seven_days(_cursor, _conn)
    day_list = helper.last_seven_day_in_int()
    day_list.reverse()

    for day in day_list:
        daily_total_win[day] = sum(
            1 for item in trades_list if ((item.calculate_profit_ratio()['roi'] > 0) & (datetime.strptime(item.updated_at, "%Y-%m-%d %H:%M:%S.%f").day == day)))
        daily_total_lose[day] = sum(
            1 for item in trades_list if ((item.calculate_profit_ratio()['roi'] < 0) & (datetime.strptime(item.updated_at, "%Y-%m-%d %H:%M:%S.%f").day == day)))
        daily_total_trades[day] = daily_total_win[day] + daily_total_lose[day]
        daily_profit[day] = sum(
            item.calculate_profit_ratio()['pnl'] for item in trades_list if (datetime.strptime(item.updated_at, "%Y-%m-%d %H:%M:%S.%f").day == day))

    table = PrettyTable()
    table.field_names = ["Days", "Profit", "Win", "Loss"]

    for day in day_list:
        date = day
        profit = daily_profit[day] if daily_total_trades[day] > 0 else "-"
        win = daily_total_win[day] if daily_total_trades[day] > 0 else "-"
        loss = daily_total_lose[day] if daily_total_trades[day] > 0 else "-"

        table.add_row([date, profit, win, loss])

    table.align = "r"
    table_string = '```\n{}```'.format(table.get_string())
    message_output = table_string
    return message_output, ParseMode.MARKDOWN


def command_balance(message: Message):
    _config = temp_storage_data[TempStorage.config]
    _exchange: ccxt.binance = temp_storage_data[TempStorage.exchange]
    _cursor = temp_storage_data[TempStorage.cursor]
    _conn = temp_storage_data[TempStorage.conn]
    starting_balance = 0
    total_balance = 0
    trades_list = _find_completed_trade(_cursor, _conn)
    profit_closed_pnl = sum(item.calculate_profit_ratio()[
        'pnl'] for item in trades_list)
    profit_closed_roi = sum(item.calculate_profit_ratio()[
                            'roi'] for item in trades_list)

    if (_config['dry_run'] == True):
        starting_balance = _config['dry_run_wallet']
        total_balance = starting_balance + profit_closed_pnl
    else:
        exchange_balance = _exchange.fetch_balance()
        total_balance = exchange_balance['total']['USDT']
        starting_balance = total_balance - profit_closed_pnl

    message_output = (
        f"*Starting Balance : * `{starting_balance}`\n"
        f"*Total PNL : * `{profit_closed_pnl}`\n"
        f"*Total ROI : * `{profit_closed_roi}`\n"
        f"*Total Balance : * `{total_balance}`\n"
    )
    return message_output, ParseMode.MARKDOWN


def command_start(message: Message):
    temp_storage_data[TempStorage.command_for_run] = "start"
    print("HELLO")
    message_output = f"Hello, {hbold(message.from_user.full_name)} , I am started already."
    return message_output, ParseMode.HTML


def command_stop(message: Message):
    temp_storage_data[TempStorage.command_for_run] = "stop"
    print("NOT HELLO")
    message_output = f"Hello, {hbold(message.from_user.full_name)} , I am stopping now."
    return message_output, ParseMode.HTML


def command_config():
    config = temp_storage_data[TempStorage.config]
    message_output = (
        f"*TimeFrame : * `{config['timeframe']}`\n"
        f"*Leverage : * `{config['trade_params']['leverage']}`\n"
        f"*Stake Ammount : * `{config['trade_params']['stake_ammount']}`\n"
        f"*Dry Run : * `{config['dry_run']}`\n"
    )
    return message_output, ParseMode.MARKDOWN


def command_unsupported(message: Message):
    message_output = f"{message.text} is not supported. Type '/help' to see supported commands."
    return message_output, ParseMode.HTML


def command_help(message: Message):
    return (
        "_Bot Control_\n"
        "------------\n"
        "*/start:* `Starts the trader`\n"
        "*/stop:* `Stops the trader`\n"
        "*/status:* `List of current trades`\n"
        "*/config:* `Config for this bot`\n"
        "*/balance:* `Show account balance of future accounts`\n"
        "*/profit:* `Lists cumulative profit from all finished trades.`\n"
        "*/daily:* `Shows profit or loss per day, over the last 7 days`\n"
        "*/help:* `This help message`\n"
    ), ParseMode.MARKDOWN
