# config/db_config.py

DB_URL   = "postgresql://postgres:lamiaa123@localhost:5432/test"
SCHEMA   = "ml_schema"
TABLE    = "obt"

# Pour tester en local sans PostgreSQL → utiliser le CSV directement
CSV_PATH = "avito_clean_20260508_105321.csv"