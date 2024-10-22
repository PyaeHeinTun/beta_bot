from asyncio import Queue
import sqlite3


def connectDB():
    # Connect to the SQLite database (create it if it doesn't exist)
    conn = sqlite3.connect('beta_bot.sqlite3.db')

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    # Create a table
    cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY,
                        entry_price REAL,
                        exit_price REAL,
                        is_short INTEGER,
                        created_at TEXT,
                        updated_at TEXT,
                        leverage INTEGER,
                        stake_ammount REAL,
                        pair TEXT,
                        is_completed INTEGER,
                        p_min REAL,
                        p_max REAL,
                        signal TEXT
                    )''')

    return cursor, conn


def closeDB(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    cursor.close()
    conn.close()
