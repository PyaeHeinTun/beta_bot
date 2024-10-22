from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.types import Message
from beta_bot.rpc import base as rpc
from datetime import datetime
from beta_bot.temp_storage import TempStorage, temp_storage_data


dp = Dispatcher()


@dp.message(lambda message: message.text.startswith("/"))
async def command_start_handler(message: Message) -> None:
    command = message.text.replace("/", "")
    parse_mode = ParseMode.HTML
    output_message = ""

    if command == "start":
        output_message, parse_mode = rpc.command_start(message)
    elif command == "stop":
        output_message, parse_mode = rpc.command_stop(message)
    elif command == "profit":
        output_message, parse_mode = rpc.command_profit(message)
    elif command == "balance":
        output_message, parse_mode = rpc.command_balance(message)
    elif command == "daily":
        output_message, parse_mode = rpc.command_daily(message)
    elif command == "status":
        output_message, parse_mode = rpc.command_status()
    elif command == "config":
        output_message, parse_mode = rpc.command_config()
    elif command == "help":
        output_message, parse_mode = rpc.command_help(message)
    else:
        output_message, parse_mode = rpc.command_unsupported(message)

    await message.answer(output_message, parse_mode=parse_mode)


async def run():
    config = temp_storage_data[TempStorage.config]
    token = config["telegram"]["token"]
    chat_id = config["telegram"]["chat_id"]

    bot = Bot(token, parse_mode=ParseMode.HTML)
    temp_storage_data[TempStorage.telegramBot] = bot
    temp_storage_data[TempStorage.botStartDate] = datetime.utcnow()
    temp_storage_data[TempStorage.dp] = dp
    output_message, parse_mode = rpc.command_config()
    await bot.send_message(chat_id=chat_id, text=output_message, parse_mode=parse_mode)
    await dp.start_polling(bot)
