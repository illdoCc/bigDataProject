import os
import pandas as pd
import joblib
import json
import psycopg2
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

def predict_all_stocks_next_close(csv_path, model_dir):
    with open('date.json', 'r', encoding='UTF-8') as f:
        dates = json.load(f)


    feature_cols = ['收盤價', '開盤價', '最高價', '最低價', '報酬率']
    df = pd.read_csv(csv_path, encoding='big5')
    df['stock_code'] = df['股票代碼'].astype(str)
    df['日期'] = pd.to_datetime(df['日期'])

    result = {}
    stock_list = df['stock_code'].unique()
    for i in range(1, 31):
        daily_result = {}
        date = dates[str(i)]
        date = pd.to_datetime(date)

        for stock_code in stock_list:
            sub_df = df[(df['stock_code'] == stock_code) & (df['日期'] == date)]
            if sub_df.empty:
                continue
            X = sub_df[feature_cols].values

            model_path = os.path.join(model_dir, f"xgb_model_{stock_code}.pkl")
            if not os.path.exists(model_path):
                continue
            xgb_model = joblib.load(model_path)

            # 如果沒有 scaler 檔案，自己 fit 一個
            hist_df = df[df['stock_code'] == stock_code]  # 全部歷史
            scaler = StandardScaler().fit(hist_df[feature_cols].values)
            X_scaled = scaler.transform(X)

            y_pred = xgb_model.predict(X_scaled)
            daily_result[stock_code] = float(y_pred[0])
        result[i] = daily_result
    return result



load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "8080")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
csv_path = "price.csv"
model_dir = "./models/xgb"
preds = predict_all_stocks_next_close(csv_path, model_dir)

# 連線資料庫
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
conn.autocommit = True
cur = conn.cursor()

for i, j, in preds.items():
    cur.execute(
        "INSERT INTO xgb (day, predict_prices) VALUES (%s, %s);",
        (i, json.dumps(j))
    )
cur.close()
conn.close()