import sqlite3
from datetime import datetime
from beta_bot.model.base import Trade

# id INTEGER PRIMARY KEY,
# entry_price REAL,
# exit_price REAL,
# is_short INTEGER,
# created_at TEXT,
# updated_at TEXT,
# leverage INTEGER,
# stake_ammount REAL,
# pair TEXT,
# is_completed INTEGER


def create_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection, entry_price, exit_price, is_short, created_at, updated_at, leverage, stake_ammount, pair, is_completed, p_min, p_max, signal):
    cursor.execute("INSERT INTO trades (entry_price, exit_price,is_short,created_at,updated_at,leverage,stake_ammount,pair,is_completed,p_min,p_max,signal) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (entry_price, exit_price, is_short, created_at, updated_at, leverage, stake_ammount, pair, is_completed, p_min, p_max, signal,))
    conn.commit()


def find_completed_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    cursor.execute("SELECT * FROM trades WHERE is_completed=1")
    rows = cursor.fetchall()
    return rows


def find_last_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    cursor.execute("SELECT * FROM trades ORDER BY created_at DESC LIMIT 1")
    rows = cursor.fetchall()
    return rows


def update_pending_to_completed_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection, id):
    cursor.execute(
        "UPDATE trades SET is_completed = ?, updated_at = ? WHERE id = ?",
        (1, str(datetime.utcnow()), id)
    )
    conn.commit()
    return


def update_trade_current_price(cursor: sqlite3.Cursor, conn: sqlite3.Connection, current_price: float,trade:Trade):
    cursor.execute(
        "UPDATE trades SET exit_price=? WHERE id = ?", (current_price,trade.id))
    conn.commit()


def delete_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection, id):
    cursor.execute("DELETE FROM trades WHERE id = ?", (id,))
    conn.commit()


def find_current_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    cursor.execute("SELECT * FROM trades WHERE is_completed=0")
    trades_list = cursor.fetchall()
    return trades_list


def map_tuple_into_trade(data: tuple):
    return Trade(
        trade_id=data[0],
        entry_price=data[1],
        exit_price=data[2],
        is_short=data[3],
        created_at=data[4],
        updated_at=data[5],
        leverage=data[6],
        stake_ammount=data[7],
        pair=data[8],
        is_completed=data[9],
        p_min=data[10],
        p_max=data[11],
        signal=data[12]
    )
