import numpy as np
import pandas as pd

player_cash=0 #input
model_predict_price=[] #input
close_price=[] #input
holdings=[] #input
tickers=[] #input
df=pd.DataFrame({
    'ticker':tickers,
    'close_price':close_price,
    'predicted_price':model_predict_price,
    'qty':holdings
})
output = pd.Series(0, index=df['ticker']) #the final number of stocks
buy_threshold = 0.02
sell_threshold = -0.02
portfolio = df['qty'].to_dict()
df['pred_return'] = (df['predicted_price'] / df['close_price']) - 1 #calculate the rate of return

# ========== selling ==========
sell_df = df[df['pred_return'] < sell_threshold].copy()
total_neg = sell_df['pred_return'].abs().sum()

if total_neg > 0:
    for _, row in sell_df.iterrows():
        ticker = row['ticker']
        if portfolio.get(ticker, 0) > 0 and ticker in output:
            weight = abs(row['pred_return']) / total_neg
            sell_qty = int(portfolio[ticker] * weight)
            output[ticker] -= sell_qty
            sell_price = row['close_price']
            proceeds = sell_qty * sell_price
            player_cash += proceeds #after selling stock,player can get money to buy another stock
            # portfolio[ticker] -= sell_qty

# ========== buying ==========
buy_df = df[df['pred_return'] > buy_threshold].copy()
total_pos = buy_df['pred_return'].sum()

if total_pos > 0 and player_cash > 0:
    for _, row in buy_df.iterrows():
        ticker = row['ticker']
        if ticker in output:
            weight = row['pred_return'] / total_pos
            alloc_cash = player_cash * weight
            buy_price = row['close_price']
            buy_qty = int(alloc_cash // buy_price)
            output[ticker]+=buy_qty