import sqlite3
from contextlib import contextmanager

# Define database path
DB_PATH = "/app/translations.db"

@contextmanager
def get_db_connection():
    """
    Provide a context manager for SQLite database connections.
    Yields a connection object and ensures it is properly closed.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        yield conn
    except sqlite3.Error as e:
        print(f"Database connection error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def execute_query(query, params=(), fetch=False):
    """
    Execute a SQL query with optional parameters.
    Args:
        query (str): SQL query to execute.
        params (tuple): Parameters for the query.
        fetch (bool): If True, fetch results (for SELECT queries).
    Returns:
        List of rows for SELECT queries if fetch=True, else None.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        if fetch:
            return cursor.fetchall()
        conn.commit()
        return cursor.lastrowid if query.strip().upper().startswith("INSERT") else None