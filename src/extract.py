# src/extract.py

import pandas as pd
import sqlalchemy
from config.db_config import DB_URL, SCHEMA, TABLE, CSV_PATH


def extract_obt():
    """Extraction depuis PostgreSQL → ml_schema.obt"""
    engine = sqlalchemy.create_engine(DB_URL)
    query  = f"SELECT * FROM {SCHEMA}.{TABLE}"
    df     = pd.read_sql(query, engine)
    print(f"✅ PostgreSQL — {df.shape[0]} lignes | {df.shape[1]} colonnes")
    return df


def extract_csv():
    """Extraction depuis le fichier CSV (test local sans PostgreSQL)"""
    df = pd.read_csv(CSV_PATH)
    print(f"✅ CSV chargé — {df.shape[0]} lignes | {df.shape[1]} colonnes")
    return df