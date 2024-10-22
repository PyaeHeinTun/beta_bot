from beta_bot.bot_start import bot_start
import asyncio
from beta_bot.rpc import base as rpc
from beta_bot.helper import base as helper
from beta_bot.temp_storage import TempStorage, temp_storage_data


async def main():
    config = helper.read_config()
    temp_storage_data[TempStorage.config] = config
    telegramEnabled = config["telegram"]["enabled"]
    task_list = []

    if telegramEnabled:
        rpc_task = asyncio.create_task(rpc.run())
        task_list.append(rpc_task)

    bot_task = asyncio.create_task(bot_start())
    task_list.append(bot_task)

    done, pending = await asyncio.wait(task_list, return_when=asyncio.FIRST_COMPLETED)

    for task in pending:
        task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
