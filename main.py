from fastapi import FastAPI
import psycopg2
import pandas as pd
import json
from datetime import datetime

app = FastAPI()

DB_HOST = "localhost"
DB_PORT = "8080"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "admin"

@app.post("/start")
def clear_and_load_stock():
    try:
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

        # 清空兩個表格
        cur.execute("TRUNCATE TABLE history RESTART IDENTITY CASCADE;")
        cur.execute("TRUNCATE TABLE stock RESTART IDENTITY CASCADE;")
            
        df = pd.read_csv("stock.csv", encoding='big5')
        ids = df.loc[:, "股票代碼"].drop_duplicates(ignore_index=True)
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 刷新stock table
        init_prices = {}
        holdings = {}
        for id in ids:
            # 取30天
            start_date = datetime.strptime('2021/2/25', '%Y/%m/%d')
            end_date = start_date + pd.Timedelta(days=40)

            filtered = df[(df['股票代碼'] == id) & 
                        (df['日期'] >= start_date) & 
                        (df['日期'] <= end_date)]
            
            prices_list = filtered['收盤價'].tolist()
            init_prices[id] = prices_list[0]
            holdings[id] = 0
            # 插入資料
            cur.execute(
                "INSERT INTO stock (id, history_price) VALUES (%s, %s);",
                (id, prices_list)
            )
        a = str(holdings)
        # init history
        user_list = ['player', 'al_lstm', 'al_arima', 'al_RNN', 'al_RF', 'al_XGB', 'llm_lstm', 'llm_arima', 'llm_RNN', 'llm_RF', 'llm_lstm', 'no_buy', 'RSP']    
        for user in user_list:
            cur.execute(
                "INSERT INTO history (day, user_name, holdings, cash) VALUES (%s, %s, %s, %s);",
                (1, user, json.dumps(holdings), 5000000.0)
            )

        cur.close()
        conn.close()
        return {
            "initial price": init_prices, 
            "holding": 5000000
        }
    except Exception as e:
        return {"error": str(e)}
    

@app.post("/advance")
def next_day_game():
    try:
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

        # 清空兩個表格
        cur.execute("TRUNCATE TABLE history RESTART IDENTITY CASCADE;")
        cur.execute("TRUNCATE TABLE stock RESTART IDENTITY CASCADE;")

        # RSP means Regular Saving Plan
        user_list = ['player', 'al_lstm', 'al_arima', 'al_RNN', 'al_RF', 'al_XGB', 'llm_lstm', 'llm_arima', 'llm_RNN', 'llm_RF', 'llm_lstm', 'no_buy', 'RSP']    
        df = pd.read_csv("stock.csv", encoding='big5')
        ids = df.loc[:, "股票代碼"].drop_duplicates(ignore_index=True)
        df['日期'] = pd.to_datetime(df['日期'])
        
        init_prices = {}
        for id in ids:
            # 取30天
            start_date = datetime.strptime('2021/2/25', '%Y/%m/%d')
            end_date = start_date + pd.Timedelta(days=40)

            filtered = df[(df['股票代碼'] == id) & 
                        (df['日期'] >= start_date) & 
                        (df['日期'] <= end_date)]
            
            prices_list = filtered['收盤價'].tolist()
            init_prices[id] = prices_list[0]
            # 插入資料
            cur.execute(
                "INSERT INTO stock (id, history_price) VALUES (%s, %s);",
                (id, prices_list)
            )

        cur.close()
        conn.close()
        return {
            "initial price": init_prices, 
            "holding": 5000000
        }
    except Exception as e:
        return {"error": str(e)}
    

def model_buy_or_sell():
    pass