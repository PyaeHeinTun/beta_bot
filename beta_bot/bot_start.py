from beta_bot.controller import base as controller


async def bot_start():
    await controller.fetch_multiple_coins()
