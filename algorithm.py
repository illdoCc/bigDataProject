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

def optimize_portfolio_dict(
    current_price,         # dict: {stockID: current_price}
    predicted_prices,      # dict: {stockID: predicted_price}
    current_holding_array, # list of dicts: [{"stock_id": db_index, "stock_name": "1101 台泥", "count": int}, ...]
    total_capital=1_000_000,
    price_csv_path="price.csv"
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
    
    # === build optimizer ===
    ef = EfficientFrontier(mu, S)

    # 條件一：所有股票最多 100%（表示限制操作金額，不是全資產）
    ef.add_constraint(lambda w: w <= 1.0)  # 每支最多佔 100% 的「這一成資金」

    # 條件二：台積電佔這 10% 資金中的至少 30%
    ticker_to_index = {ticker: i for i, ticker in enumerate(common_tickers)}
    if '2330' in ticker_to_index:
        ef.add_constraint(lambda w: w[ticker_to_index['2330']] >= 0.3)  # 30% of this portion

    # 最大 Sharpe 配置
    ef.max_sharpe()
    weights = ef.clean_weights(cutoff=0.001)
    
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


# load_dotenv()

# DB_HOST = os.getenv("DB_HOST", "localhost")
# DB_PORT = os.getenv("DB_PORT", "8080")
# DB_NAME = os.getenv("DB_NAME", "postgres")
# DB_USER = os.getenv("DB_USER", "postgres")
# DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")

# conn = psycopg2.connect(
#     host=DB_HOST,
#     port=DB_PORT,
#     dbname=DB_NAME,
#     user=DB_USER,
#     password=DB_PASSWORD
# )
# cur = conn.cursor()
# cur.execute("SELECT * FROM history WHERE user_name = 'al_RF';")
# row = cur.fetchone()

# day, user_name, holdings, cash = row

# holdings = json.loads(holdings)
# predicted_price = json.loads('{"1101": 52.500927582712585, "1102": 52.263799709332496, "1216": 74.82031728452469, "1301": 113.45824254106843, "1303": 91.37202255365607, "1326": 91.78288736011767, "1402": 32.59651552293965, "1476": 645.1849447362724, "1504": 34.44245820967959, "1519": 54.85612162627536, "1590": 1129.0836122684418, "2002": 39.779036921780765, "2027": 57.91578754595349, "2059": 494.0228958495089, "2105": 55.60239404522204, "2207": 650.1515380554492, "2301": 68.73421155659636, "2303": 65.16940535739205, "2308": 308.7640633665343, "2317": 127.21877324045685, "2324": 26.94690676591975, "2327": 590.305746145552, "2330": 649.4962764781731, "2345": 337.273154533811, "2347": 76.87126872035229, "2353": 33.67278648359093, "2356": 27.05603511986992, "2357": 377.7489036108971, "2360": 217.35612910163894, "2368": 96.01872831199617, "2371": 35.74782433742987, "2376": 153.25668212941852, "2377": 182.90220378506925, "2379": 567.6405310707584, "2382": 97.58587784943376, "2383": 276.1689581250775, "2385": 100.4862876680963, "2395": 392.90114262792275, "2408": 96.51754215650949, "2409": 30.62085086877991, "2412": 129.99684783094125, "2449": 47.38122248569048, "2454": 1120.3273036160904, "2474": 205.4329941535018, "2542": 48.72853261344685, "2603": 199.24380831668327, "2609": 200.74983563093684, "2610": 28.30693102580119, "2615": 287.63362972883596, "2618": 34.261618279899714, "2801": 19.13414908137317, "2812": 15.542679338341065, "2834": 13.34329046103788, "2880": 23.93443155144053, "2881": 81.74036316127906, "2882": 63.87319902009816, "2883": 19.1420292887821, "2884": 34.08678105604856, "2885": 26.440970573153955, "2886": 43.744131208285246, "2887": 20.110871028552257, "2888": 11.407864079370336, "2890": 18.260594376272685, "2891": 29.486880425278752, "2892": 28.933273748225034, "2912": 290.48860820259597, "3008": 3385.0536794933914, "3017": 115.64451149905842, "3034": 588.3783742994309, "3036": 87.9482321706108, "3037": 240.0490863088418, "3044": 136.94135566488993, "3045": 108.1006731333549, "3231": 33.367016753977005, "3443": 828.8443593707899, "3481": 27.190572845687473, "3533": 876.0458218473655, "3653": 426.03943411315987, "3661": 1171.8339753890477, "3665": 325.85787990396466, "3702": 56.388355118725165, "3711": 125.2017394524177, "4904": 85.19466789343942, "4938": 77.85914835306453, "4958": 123.9788837924346, "5269": 2073.91057756056, "5871": 264.4955421771549, "5876": 52.533616044969484, "5880": 29.2401129716759, "6409": 1743.4646790211316, "6415": 4682.196671796367, "6446": 562.0252851310406, "6505": 108.47382437136771, "6669": 1080.1601061405643, "8464": 457.8187690018835, "9904": 37.96172935675067, "9910": 236.39482984347939, "9945": 77.81217702065294}')
# current_price = json.loads('{"1101": 52.500927582712585, "1102": 52.263799709332496, "1216": 74.8287865409992, "1301": 113.45824254106843, "1303": 91.37202255365607, "1326": 91.78288736011767, "1402": 32.56608260068303, "1476": 645.1849447362724, "1504": 34.44245820967959, "1519": 54.85612162627536, "1590": 1129.0836122684418, "2002": 39.76681222542449, "2027": 57.91578754595349, "2059": 494.0228958495089, "2105": 55.60239404522204, "2207": 650.1515380554492, "2301": 68.73421155659636, "2303": 65.16940535739205, "2308": 308.7640633665343, "2317": 127.21877324045685, "2324": 26.94690676591975, "2327": 590.305746145552, "2330": 650.3340438635391, "2345": 337.273154533811, "2347": 76.87126872035229, "2353": 33.67278648359093, "2356": 27.05603511986992, "2357": 377.7489036108971, "2360": 217.35612910163894, "2368": 96.01872831199617, "2371": 35.74782433742987, "2376": 153.25668212941852, "2377": 182.90220378506925, "2379": 567.6405310707584, "2382": 97.58587784943376, "2383": 276.1689581250775, "2385": 101.90987525162753, "2395": 392.90114262792275, "2408": 96.51754215650949, "2409": 30.62085086877991, "2412": 129.99684783094125, "2449": 47.38122248569048, "2454": 1120.3273036160904, "2474": 205.4329941535018, "2542": 48.72853261344685, "2603": 199.24380831668327, "2609": 200.74983563093684, "2610": 28.30693102580119, "2615": 287.63362972883596, "2618": 34.261618279899714, "2801": 19.13414908137317, "2812": 15.542679338341065, "2834": 13.34329046103788, "2880": 23.93443155144053, "2881": 81.74036316127906, "2882": 63.87319902009816, "2883": 19.1420292887821, "2884": 34.08678105604856, "2885": 26.440970573153955, "2886": 43.744131208285246, "2887": 20.110871028552257, "2888": 11.407864079370336, "2890": 18.260594376272685, "2891": 29.486880425278752, "2892": 28.933273748225034, "2912": 290.48860820259597, "3008": 3385.0536794933914, "3017": 115.64451149905842, "3034": 588.3783742994309, "3036": 87.9482321706108, "3037": 239.83282793780106, "3044": 136.94135566488993, "3045": 108.1006731333549, "3231": 33.367016753977005, "3443": 828.8443593707899, "3481": 27.190572845687473, "3533": 876.0458218473655, "3653": 426.03943411315987, "3661": 1171.8339753890477, "3665": 324.73490068068037, "3702": 56.388355118725165, "3711": 125.2017394524177, "4904": 85.19466789343942, "4938": 77.85914835306453, "4958": 123.9788837924346, "5269": 2073.91057756056, "5871": 264.4955421771549, "5876": 52.533616044969484, "5880": 29.2401129716759, "6409": 1743.4646790211316, "6415": 4682.196671796367, "6446": 562.0252851310406, "6505": 108.47382437136771, "6669": 1080.5736775691357, "8464": 457.8187690018835, "9904": 37.96172935675067, "9910": 236.39482984347939, "9945": 77.81217702065294}')

# cur.close()
# conn.close()

# result = optimize_portfolio_dict(
#     current_price, predicted_price, holdings,
#     total_capital=5_000_000 * 0.1,
#     price_csv_path="price.csv"
# )

# print(result)
