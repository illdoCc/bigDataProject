from fastapi import FastAPI, HTTPException
import psycopg2
import pandas as pd
import json
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
from dotenv import load_dotenv
import algorithm
import os

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

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "8080")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")



with open('predict_price/price_labels.json', 'r', encoding='UTF-8') as f:
    price_labels = json.load(f)

with open('predict_price/rf_predict_prices.json', 'r', encoding='UTF-8') as f:
    rf_predict_prices = json.load(f)
with open('predict_price/lstm_predict_prices.json', 'r', encoding='UTF-8') as f:
    lstm_predict_prices = json.load(f)
with open('predict_price/rnn_predict_prices.json', 'r', encoding='UTF-8') as f:
    rnn_predict_prices = json.load(f)
with open('predict_price/xgb_predict_prices.json', 'r', encoding='UTF-8') as f:
    xgb_predict_prices = json.load(f)
with open('predict_price/arima_predict_prices.json', 'r', encoding='UTF-8') as f:
    arima_predict_prices = json.load(f)

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

        df = pd.read_csv("price.csv", encoding='big5')
        ids = df.loc[:, "股票代碼"].drop_duplicates(ignore_index=True)
        df['日期'] = pd.to_datetime(df['日期'])

        with open('stock_name.json', 'r', encoding='UTF-8') as f:
            stock_name = json.load(f)

        # 刷新stock table
        stock_init_prices = [] # return value
        holdings = []
        for id, stock_symbol in enumerate(ids):
            # 取30天
            start_date = datetime.strptime('2022/8/30', '%Y/%m/%d')
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
        user_list = ['player', 'al_lstm', 'al_arima', 'al_RNN', 'al_RF', 'al_XGB', 'no_buy']
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
def next_day_game(day:int, portfolio: Portfolio):
    # day -= 1
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

        # only updates the history with history of previous day
        # -2 because database starts from day 0, but frontend starts from day 1
        cur.execute("SELECT * FROM history WHERE day=%s",(day-2, ))
        rows = cur.fetchall()
        
        for row in rows:
            day, user_name, holdings, cash = row
            day += 1
            # print(day)
            if user_name == 'no_buy':
                histories.append({
                    "day": day,
                    "user_name": "no_buy",
                    "holdings": json.loads(holdings),
                    "cash": cash
                })
            elif user_name != 'player':
                model = user_name.split('_')[1]
                histories.append(model_buy_or_sell(day, model, json.loads(holdings), cash))
                
            elif user_name == 'player':
                histories.append(player_portfolio(cur, day, portfolio))
        
        
        records = [
            (h["day"], h["user_name"], json.dumps(h["holdings"]), h["cash"])
            for h in histories
        ]

        cur.executemany(
            "INSERT INTO history (day, user_name, holdings, cash) VALUES (%s, %s, %s, %s);",
            records
        )


        cur.close()
        conn.close()
        return {
            "stocks": latest_stock_price,
            "histories": histories
        }
    except Exception as e:
        tb = traceback.format_exc()
        return {"status": "error", "message": str(e), "traceback": tb}
        raise HTTPException(status_code=400, detail=str(e))
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

def player_portfolio(cur, day:int, portfolio):
    portfolio = portfolio.stocks
    cur.execute("SELECT * FROM history WHERE user_name = %s AND day= %s", ("player",str(day-1)))
    player_info = cur.fetchone()
    _, user_name, holdings, cash = player_info
    holdings = json.loads(holdings)

    # 一次性查詢所有股票資料
    stock_ids = [int(portfo["id"]) for portfo in portfolio]
    placeholders = ', '.join(['%s'] * len(stock_ids))
    cur.execute(f"SELECT * FROM stock WHERE id IN ({placeholders})", tuple(stock_ids))
    stocks = {row[0]: row[1:] for row in cur.fetchall()}

    # 更新持有量和現金
    for portfo in portfolio:
        stock = stocks[int(portfo["id"])]
        history_prices = stock[0]
        # buy_or_sell_price = history_prices[day - 1]
        buy_or_sell_amount = portfo["count"]

        holdings[int(portfo["id"])]["count"] += buy_or_sell_amount
        if holdings[int(portfo["id"])]["count"] < 0:
            raise HTTPException(status_code=400, detail="You do not have enough shares to sell.")
        
        cash -= (history_prices[day]) * buy_or_sell_amount
        cash = max(cash, 0)

    return {
        "day": day,
        "user_name": "player",
        "holdings": holdings,
        "cash": cash
    }



# 依據預測價格決定買與賣多少
def model_buy_or_sell(day, model, holdings, cash):
    print("TEST1")
    current_price = {stock_id: values[day] for stock_id, values in price_labels.items() if len(values) > day}
    if model == 'RF':
        predicted_prices=rf_predict_prices[str(day)]
    elif model == "XGB":
        predicted_prices=xgb_predict_prices[str(day)]
    elif model == "lstm":
        predicted_prices=lstm_predict_prices[str(day)]
    elif model == 'RNN':
        predicted_prices=rnn_predict_prices[str(day)]
    elif model == 'arima':
        predicted_prices=arima_predict_prices[str(day)]

    portfolio = algorithm.optimize_portfolio_dict(
        current_price=current_price, 
        predicted_prices=predicted_prices,
        current_holding_array=holdings,
        total_capital=cash*0.1
    )

    print("TEST2")
    
    for stock, count in portfolio.items():
        print(stock, count)
        history_price = price_labels[stock][day]  # 取得當天該股票價格
        buy_or_sell_amount = count

        # 根據股票代碼找到對應 holdings 中的那筆資料（原始 list 會被直接修改）
        holding = next((item for item in holdings if item["stock_name"].startswith(stock)), None)

        if holding is None:
            raise ValueError(f"找不到股票代碼 {stock} 對應的持股資料")

        transaction_amount = buy_or_sell_amount * history_price

        if buy_or_sell_amount > 0:
            # 買進：增加持股，減少現金
            holding["count"] += buy_or_sell_amount
            cash -= transaction_amount
        elif buy_or_sell_amount < 0:
            # 賣出：檢查是否持有足夠股票
            if holding["count"] < abs(buy_or_sell_amount):
                raise ValueError(f"股票 {stock} 賣出失敗，持有數量不足")
            holding["count"] += buy_or_sell_amount  # 負值等同於減少
            cash += abs(transaction_amount)

    return {
        "day": day,
        "user_name": f"al_{model}",
        "holdings": holdings,
        "cash": cash
    }    


def get_single_stock_subset(df, stock_id, start_date_str, num_records):
    """
    從指定股票代碼中，擷取從起始日開始的連續 num_records 筆交易日資料。
    
    參數：
    - df: 整份股票資料（包含 '股票代碼' 和 '日期' 欄）
    - stock_id: 股票代碼（如 1101）
    - start_date_str: 起始日期（字串格式，如 '2020/1/3'）
    - num_records: 要擷取幾筆資料（從起始日含當日開始）

    回傳：
    - 該股票的子 DataFrame（如果找到且筆數足夠）
    - 若無資料或資料不足則回傳 None，並印出警告
    """
    # 轉換日期欄
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    start_date = pd.to_datetime(start_date_str)

    # 篩選該股票
    stock_df = df[df['股票代碼'] == stock_id].sort_values('日期').reset_index(drop=True)

    if stock_df.empty:
        print(f"⚠ 找不到股票代碼 {stock_id} 的資料")
        return None

    # 找起始日期的位置
    start_idx_list = stock_df.index[stock_df['日期'] == start_date].tolist()
    if not start_idx_list:
        print(f"⚠ 股票 {stock_id} 找不到起始日 {start_date.date()}")
        return None

    start_idx = start_idx_list[0]
    sub_df = stock_df.iloc[start_idx:start_idx + num_records]

    if len(sub_df) < num_records:
        print(f"⚠ 股票 {stock_id} 起始日後不足 {num_records} 筆資料")
        return None

    return sub_df