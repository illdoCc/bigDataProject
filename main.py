from fastapi import FastAPI, HTTPException
import psycopg2
import pandas as pd
import json
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

app = FastAPI()

# cors setting
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173", 
    "https://stock.leowang.dev"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # 允許的來源
    allow_credentials=True,
    allow_methods=["*"],              # 允許的 HTTP 方法 (GET, POST...)
    allow_headers=["*"],              # 允許的 HTTP 標頭
)

DB_HOST = "localhost"
DB_PORT = "8080"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "admin"

@app.get("/start")
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

        with open('stock_name.json', 'r', encoding='UTF-8') as f:
            stock_name = json.load(f)
        
        # 刷新stock table
        stock_init_prices = [] # return value
        holdings = []
        for id, stock_symbol in enumerate(ids):
            # 取30天
            start_date = datetime.strptime('2021/2/25', '%Y/%m/%d')
            end_date = start_date + pd.Timedelta(days=60)
            filtered = df[(df['股票代碼'] == stock_symbol) & 
                        (df['日期'] >= start_date) & 
                        (df['日期'] <= end_date)]
            

            prices_list = filtered['收盤價'].round(2).tolist()
            stock_init_prices.append({
                "id": id,
                "name": f"{stock_symbol} {stock_name.get(str(stock_symbol), '')}",
                "price": prices_list[0]
            })
            holdings.append({
                "stock_id": id,
                "stock_name": f"{stock_symbol} {stock_name.get(str(stock_symbol), '')}",
                "count": 0
            })
            # holdings[stock_symbol] = 0
            # 插入資料
            cur.execute(
                "INSERT INTO stock (id, history_price, name, stock_symbol) VALUES (%s, %s, %s, %s);",
                (id, prices_list, stock_name.get(str(stock_symbol), ""), stock_symbol)
            )


        histories = [] # return value
        user_list = ['player', 'al_lstm', 'al_arima', 'al_RNN', 'al_RF', 'al_XGB', 'llm_lstm', 'llm_arima', 'llm_RNN', 'llm_RF', 'llm_lstm', 'no_buy']    
        for user in user_list:
            cur.execute(
                "INSERT INTO history (day, user_name, holdings, cash) VALUES (%s, %s, %s, %s);",
                (0, user, json.dumps(holdings), 5000000.0)
            )
        
            histories.append({
                "day": 1,
                "user_name": user,
                "holdings": holdings,
                "cash": 5000000
            })

        cur.close()
        conn.close()
        return {
            "stocks": stock_init_prices, 
            "histories": histories
        }
    except Exception as e:
        return {"error": str(e)}
    

    

class Portfolio(BaseModel):
    stocks: list

@app.post("/advance/{day}")
def next_day_game(day, portfolio: Portfolio):
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

        latest_stock_price = get_latest_stock_price(cur, day)
        histories = []
        
        # model_list = ['arima', 'RNN', 'lstm', 'RF', 'XGB']
        # rows = cur.execute
        cur.execute("SELECT * FROM history")
        rows = cur.fetchall()
        for row in rows:
            day, user_name, holdings, cash = row
            day += 1
            
            if user_name == 'no_buy':
                cur.execute(
                    "INSERT INTO history (day, user_name, holdings, cash) VALUES (%s, %s, %s, %s);",
                    (day, user_name, json.dumps(holdings), 5000000.0)
                )
            elif user_name != 'player':
                pass
                # model = user_name.split('_')[1]
                # predict_prices = model_predict_price(model, '2021/2/25', day)
                # holdings, cash = model_buy_or_sell(predict_prices, holdings, cash)
                # cur.execute(
                #     "INSERT INTO history (day, user_name, holdings, cash) VALUES (%s, %s, %s, %s);",
                #     (day, user_name, json.dumps(holdings), cash)
                # )
            elif user_name == 'player':
                histories.append(player_portfolio(cur, day, portfolio))
            
        for history in histories:
            cur.execute(
                "INSERT INTO history (day, user_name, holdings, cash) VALUES (%s, %s, %s, %s);",
                (history["day"], history["user_name"], json.dumps(history["holdings"]), history["cash"])
            )
                
 
        cur.close()
        conn.close()
        return {
            "stocks": latest_stock_price, 
            "histories": histories
        }
    except Exception as e:
        return {"error": traceback.format_exc()}
# end of next_day_game


# 取得當天股價
def get_latest_stock_price(cur, day):
    day = int(day)
    latest_stock_price = []

    cur.execute("SELECT * FROM stock")
    rows = cur.fetchall()
    for row in rows:
        id, history_prices, name, stock_symbol = row
        latest_stock_price.append({
            "id": id,
            "name": f"{stock_symbol} {name}",
            "price": history_prices[day]
        })

    return latest_stock_price
# end of get_latest_stock_price

def player_portfolio(cur, day, portfolio):
    portfolio = portfolio.stocks
    day = int(day)
    cur.execute("SELECT * FROM history WHERE user_name = %s", ("player",))
    player_info = cur.fetchone()
    _, user_name, holdings, cash = player_info
    holdings = json.loads(holdings)

    for portfo in portfolio:
        cur.execute("SELECT * FROM stock WHERE id = %s", (int(portfo["id"]),))
        row = cur.fetchone()
        id, history_prices, name, stock_symbol = row
        buy_or_sell_price = history_prices[day - 1]
        buy_or_sell_amount = portfo["count"]

        holdings[id]["count"] += buy_or_sell_amount
        if holdings[id]["count"] < 0:
            raise HTTPException(status_code=400, detail="You do not have enough shares to sell.")
        cash += (history_prices[day] - buy_or_sell_price) * buy_or_sell_amount
        if cash < 0:
            cash = 0
        
    return {
        "day": day,
        "user_name": "player",
        "holdings": holdings,
        "cash": cash
    }

        


# 取得模型預測價格
price_label_df = pd.read_csv('price_labeled.csv', encoding='big5')
# get next day prices of all stocks
def model_predict_price(model, start_date, day):
    predict_prices = {}
    stock_models = os.listdir(model)
    start_date = datetime.strptime(start_date, '%Y/%m/%d')
    today = start_date + pd.Timedelta(days=day - 1)

    for stock in stock_models:
        stock_id = stock[:4]

    return predict_prices
# end of model_predict_price


# 依據預測價格決定買與賣多少
def model_buy_or_sell(predict_prices, holdings, cash):
    pass