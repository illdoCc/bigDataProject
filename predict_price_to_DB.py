import os
import json
import psycopg2
from dotenv import load_dotenv

with open('lstm_pred_result.json', 'r', encoding='UTF-8') as f:
    data = json.load(f)

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "8080")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
cur = conn.cursor()
for d, p in data.items():
    if d != "0" and d != "31":
        cur.execute(
            "INSERT INTO lstm (day, predict_prices) VALUES (%s, %s);",
            (int(d), json.dumps(p))
        )

conn.commit()
cur.close()
conn.close()