# db.py
import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # read the .env

def get_connection(env: str = "DEV"):
    """
    return a connection pyodbc a SQL Server.

    Parameters
    ----------
    env : str
        "DEV" o "PROD". by default "DEV" to avoid any change in PROD.
    """
    env = env.upper()
    if env not in ("DEV", "PROD"):
        raise ValueError(f"evirontment no valid: {env!r}. Use 'DEV' or 'PROD'.")

    # anti-accident: ask confirmation to PROD
    if env == "PROD":
        confirmacion = input(
            "⚠️  You are connecting to PRODUCTION. Write 'PROD' to continue: "
        )
        if confirmacion.strip() != "PROD":
            raise RuntimeError("Connection to PROD cancelled by user.")

    params = {
        "DRIVER": "{" + os.getenv("ODBC_DRIVER") + "}",
        "SERVER": os.getenv(f"{env}_SERVER"),
        "DATABASE": os.getenv(f"{env}_DATABASE"),
        "UID": os.getenv(f"{env}_UID"),
        "PWD": os.getenv(f"{env}_PWD"),
        "TrustServerCertificate": "yes",
    }

    # validation: avise if are missing a variable in .env
    missing = [k for k, v in params.items() if v in (None, "{None}")]
    if missing:
        raise KeyError(
            f"Missing variables in .env to {env}: {missing}"
        )

    conn_str = ";".join(f"{k}={v}" for k, v in params.items())

    # timeout=5 → si no logra conectar en 5s, lanza error en vez de colgarse
    conn = pyodbc.connect(conn_str, timeout=5)
    conn.timeout = 60   # las consultas fallan tras 60s en vez de esperar infinito
    return conn

def run_query(sql: str, env: str = "DEV", params: tuple | None = None):
    """
    open the connection, execute the statement, return the dataframe and close the connection
    """ 
    conn = get_connection(env)
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()