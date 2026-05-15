import os
from dotenv import load_dotenv
 
# Charger les variables du fichier .env
load_dotenv()
 
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_NAME     = os.getenv("DB_NAME")
 
# URL construite automatiquement depuis les variables .env
DB_URL  = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
SCHEMA  = "ml_schema"
TABLE   = "obt"
 
# Chemin vers le CSV (pour tester sans PostgreSQL)
CSV_PATH = "data/avito_clean_20260508_105321.csv"