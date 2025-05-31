import numpy as np
import pandas as pd
import psycopg2
from pypfopt import EfficientFrontier, risk_models
from dotenv import load_dotenv
import os
import json

# def optimize_portfolio_dict(
#     current_price,         # dict: {stockID: current_price}
#     predicted_prices,      # dict: {stockID: predicted_price}
#     current_holding_array, # list of dicts: [{"stock_id": db_index, "stock_name": "1101 台泥", "count": int}, ...]
#     total_capital=1_000_000,
#     price_csv_path="price.csv"
# ):
#     """
#     使用PyPortfolioOpt根據dict格式輸入優化持股調整

#     回傳
#     ----
#     adjustment_array: np.array         # shape=(1, len(tickers)), 每檔應增減股數
#     weights: dict                      # 每檔分配權重
#     tickers: list[str]                 # 輸出tickers順序，可對應 adjustment_array
#     """

#     # 取所有交集股票代碼（有報價有預測才納入）
#     tickers = sorted(set(current_price) & set(predicted_prices))
    
#     # 將current_holding_array轉為dict: {stockID: 持有股數}
#     holding_dict = {}
#     for item in current_holding_array:
#         # stock_name格式為 "1101 台泥"，只取 stockID
#         stock_id = item["stock_name"].split()[0]
#         holding_dict[stock_id] = item["count"]
#     # 沒出現在holding_dict裡的當0
#     current_holdings = {ticker: holding_dict.get(ticker, 0) for ticker in tickers}
    
#     # === set forecast table ===
#     df_forecast = pd.DataFrame({
#         "Ticker": tickers,
#         "CurrentPrice": [current_price[ticker] for ticker in tickers],
#         "PredictedPrice": [predicted_prices[ticker] for ticker in tickers]
#     })
#     df_forecast["expected_return"] = (
#         df_forecast["PredictedPrice"] - df_forecast["CurrentPrice"]
#     ) / df_forecast["CurrentPrice"]
#     df_forecast.set_index("Ticker", inplace=True)
    
#     # === load history price ===
#     df_raw = pd.read_csv(price_csv_path, encoding="big5")
#     df_raw["日期"] = pd.to_datetime(df_raw["日期"])
#     df_prices = df_raw.pivot(index="日期", columns="股票代碼", values="收盤價").sort_index()
#     df_prices.columns = df_prices.columns.astype(str)
    
#     # === data align ===
#     common_tickers = list(set(df_forecast.index) & set(df_prices.columns))
#     df_forecast = df_forecast.loc[common_tickers]
#     df_prices = df_prices[common_tickers]
    
#     # === calculate return and covariance matrix ===
#     mu = df_forecast["expected_return"]
#     S = risk_models.exp_cov(df_prices)
    
#     # === use Efficient Frontier to find max Sharpe  ===
#     ef = EfficientFrontier(mu, S)
#     ef.max_sharpe()
#     weights = ef.clean_weights(cutoff=0.001)
    
#     # === calculate stock adjustment ===
#     adjustment_vector = []
#     for ticker in common_tickers:
#         target_weight = weights.get(ticker, 0)
#         current_price_val = df_forecast.loc[ticker, "CurrentPrice"]
#         target_shares = (target_weight * total_capital) / current_price_val
#         current_shares = current_holdings.get(ticker, 0)
#         diff_shares = int(round(target_shares - current_shares))
#         adjustment_vector.append(diff_shares)
    
#     # === output: adjustment_array ===
#     adjustment_array = np.zeros((1, len(tickers)), dtype=int)
#     ticker_to_index = {ticker: i for i, ticker in enumerate(tickers)}
#     for i, ticker in enumerate(common_tickers):
#         if ticker in ticker_to_index:
#             idx = ticker_to_index[ticker]
#             adjustment_array[0, idx] = adjustment_vector[i]
    
#     return adjustment_array, weights

# ---------------- 使用範例 -----------------
"""
current_price = {"1101": 52.3, "2330": 839, ...}
predicted_prices = {"1101": 54.0, "2330": 860, ...}
current_holding_array = [
    {"stock_id": 1, "stock_name": "1101 台泥", "count": 100},
    {"stock_id": 2, "stock_name": "2330 台積電", "count": 50},
    # ...
]
adj_array, weights, tickers = optimize_portfolio_dict(
    current_price, predicted_prices, current_holding_array,
    total_capital=5_000_000,
    price_csv_path="price.csv"
)
# adj_array.shape = (1, len(tickers))
# 你可根據 tickers[i] 取得 adj_array[0, i] 來操作個股調整數量
"""

# def optimize_portfolio_dict(
#     current_price,         # dict: {stockID: current_price}
#     predicted_prices,      # dict: {stockID: predicted_price}
#     current_holding_array, # list of dicts: [{"stock_id": db_index, "stock_name": "1101 台泥", "count": int}, ...]
#     total_capital=1_000_000,
#     price_csv_path="price.csv"
# ):
#     import numpy as np
#     import pandas as pd
#     from pypfopt import EfficientFrontier, risk_models

#     tickers = sorted(set(current_price) & set(predicted_prices))
    
#     # 將current_holding_array轉為dict: {stockID: 持有股數}
#     holding_dict = {}
#     for item in current_holding_array:
#         stock_id = item["stock_name"].split()[0]
#         holding_dict[stock_id] = item["count"]
#     current_holdings = {ticker: holding_dict.get(ticker, 0) for ticker in tickers}
    
#     # === set forecast table ===
#     df_forecast = pd.DataFrame({
#         "Ticker": tickers,
#         "CurrentPrice": [current_price[ticker] for ticker in tickers],
#         "PredictedPrice": [predicted_prices[ticker] for ticker in tickers]
#     })
#     df_forecast["expected_return"] = (
#         df_forecast["PredictedPrice"] - df_forecast["CurrentPrice"]
#     ) / df_forecast["CurrentPrice"]
#     df_forecast.set_index("Ticker", inplace=True)
    
#     # === load history price ===
#     df_raw = pd.read_csv(price_csv_path, encoding="big5")
#     df_raw["日期"] = pd.to_datetime(df_raw["日期"])
#     df_prices = df_raw.pivot(index="日期", columns="股票代碼", values="收盤價").sort_index()
#     df_prices.columns = df_prices.columns.astype(str)
    
#     # === data align ===
#     common_tickers = list(set(df_forecast.index) & set(df_prices.columns))
#     df_forecast = df_forecast.loc[common_tickers]
#     df_prices = df_prices[common_tickers]
#     current_holdings = {ticker: current_holdings.get(ticker, 0) for ticker in common_tickers}
    
#     # === calculate return and covariance matrix ===
#     mu = df_forecast["expected_return"]
#     S = risk_models.exp_cov(df_prices)
    
#     # === build optimizer ===
#     ef = EfficientFrontier(mu, S)

#     # 條件一：所有股票最多 100%（表示限制操作金額，不是全資產）
#     ef.add_constraint(lambda w: w <= 1.0)  # 每支最多佔 100% 的「這一成資金」

#     # 條件二：台積電佔這 10% 資金中的至少 30%
#     ticker_to_index = {ticker: i for i, ticker in enumerate(common_tickers)}
#     if '2330' in ticker_to_index:
#         ef.add_constraint(lambda w: w[ticker_to_index['2330']] >= 0.3)  # 30% of this portion

#     # 最大 Sharpe 配置
#     ef.max_sharpe()
#     weights = ef.clean_weights(cutoff=0.001)
    
#  # === output: adjustment_dict ===
#     adjustment_dict = {}
#     for i, ticker in enumerate(common_tickers):
#         target_weight = weights.get(ticker, 0)
#         current_price_val = df_forecast.loc[ticker, "CurrentPrice"]
#         target_shares = (target_weight * total_capital) / current_price_val
#         current_shares = current_holdings.get(ticker, 0)
#         diff_shares = int(round(target_shares - current_shares))
#         if diff_shares != 0:
#             adjustment_dict[ticker] = diff_shares

#     return adjustment_dict

def optimize_portfolio_dict(
    current_price,         # dict: {stockID: current_price}
    predicted_prices,      # dict: {stockID: predicted_price}
    current_holding_array, # list of dicts: [{"stock_id": db_index, "stock_name": "1101 台泥", "count": int}, ...]
    total_capital=1_000_000,
    price_csv_path="price.csv",
    solver="SCS"           # 預設改用 SCS
):
    import numpy as np
    import pandas as pd
    from pypfopt import EfficientFrontier, risk_models

    tickers = sorted(set(current_price) & set(predicted_prices))

    # 將current_holding_array轉為dict: {stockID: 持有股數}
    holding_dict = {}
    for item in current_holding_array:
        stock_id = item["stock_name"].split()[0]
        holding_dict[stock_id] = item["count"]
    current_holdings = {ticker: holding_dict.get(ticker, 0) for ticker in tickers}

    # === set forecast table ===
    df_forecast = pd.DataFrame({
        "Ticker": tickers,
        "CurrentPrice": [current_price[ticker] for ticker in tickers],
        "PredictedPrice": [predicted_prices[ticker] for ticker in tickers]
    })
    df_forecast["expected_return"] = (
        df_forecast["PredictedPrice"] - df_forecast["CurrentPrice"]
    ) / df_forecast["CurrentPrice"]
    df_forecast.set_index("Ticker", inplace=True)

    # === load history price ===
    df_raw = pd.read_csv(price_csv_path, encoding="big5")
    df_raw["日期"] = pd.to_datetime(df_raw["日期"])
    df_prices = df_raw.pivot(index="日期", columns="股票代碼", values="收盤價").sort_index()
    df_prices.columns = df_prices.columns.astype(str)

    # === data align ===
    common_tickers = list(set(df_forecast.index) & set(df_prices.columns))
    df_forecast = df_forecast.loc[common_tickers]
    df_prices = df_prices[common_tickers]
    current_holdings = {ticker: current_holdings.get(ticker, 0) for ticker in common_tickers}

    # === calculate return and covariance matrix ===
    mu = df_forecast["expected_return"]
    S = risk_models.exp_cov(df_prices)

    # === Robustness: 檢查 NaN/inf ===
    if np.isnan(mu).any() or np.isnan(S.values).any() or np.isinf(mu).any() or np.isinf(S.values).any():
        print("資料有 NaN 或 inf，無法最佳化")
        return {}

    # 強化 S（避免 singular）
    epsilon = 1e-6
    S += np.eye(S.shape[0]) * epsilon

    # === build optimizer ===
    ef = EfficientFrontier(mu, S)

    # 條件一：所有股票最多 100%
    ef.add_constraint(lambda w: w <= 0.5)

    # 條件二：台積電佔這 10% 資金中的至少 30%
    ticker_to_index = {ticker: i for i, ticker in enumerate(common_tickers)}
    if '2330' in ticker_to_index:
        ef.add_constraint(lambda w: w[ticker_to_index['2330']] >= 0.3)

    # 最大 Sharpe 配置，Try/Except 防呆
    try:
        ef.max_sharpe()
        weights = ef.clean_weights(cutoff=0.001)
    except Exception as e:
        return {}

    # === output: adjustment_dict ===
    adjustment_dict = {}
    for i, ticker in enumerate(common_tickers):
        target_weight = weights.get(ticker, 0)
        current_price_val = df_forecast.loc[ticker, "CurrentPrice"]
        target_shares = (target_weight * total_capital) / current_price_val
        current_shares = current_holdings.get(ticker, 0)
        diff_shares = int(round(target_shares - current_shares))
        if diff_shares != 0:
            adjustment_dict[ticker] = diff_shares

    return adjustment_dict
