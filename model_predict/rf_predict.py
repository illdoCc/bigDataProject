import os
import pandas as pd
import joblib
import json
import pandas as pd

def predict_all_stocks_in_memory(df, all_models, feature_cols = ['收盤價', '開盤價', '最高價', '最低價', '報酬率']):
    with open('date.json', 'r', encoding='UTF-8') as f:
        dates = json.load(f)

    results = {}
    for i in range(31):
        daily_result = {}
        df['日期'] = pd.to_datetime(df['日期'])
        date = dates[str(i)]
        date = pd.to_datetime(date)
        stock_list = df['股票代碼'].unique()

        for stock in stock_list:
            sub_df = df[(df['股票代碼'] == stock) & (df['日期'] == date)]
            if sub_df.empty:
                continue
            rf_model = all_models.get(str(stock))
            if rf_model is None:
                continue
            X = sub_df[feature_cols].copy()
            try:
                pred = rf_model.predict(X)
                daily_result[int(stock)] = float(pred[0])
            except Exception as e:
                print(f"{stock} 預測失敗: {e}")

        results[i] = daily_result

    
    return results