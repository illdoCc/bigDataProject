import os
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# 參數
stock_id = 1102
sequence_length = 30
features = ['最後揭示買價', '最高價', '高低價差', '最後揭示賣價', '本益比_TEJ',
            '最低價', '開盤價', '股價淨值比_TEJ', '股利殖利率', '成交金額比重', '周轉率']
target = '收盤價'
model_path = f'1102_LSTM_model.h5'
csv_path = 'price.csv'

# 1. 讀資料
df = pd.read_csv(csv_path, encoding='big5')
df['日期'] = pd.to_datetime(df['日期'])

# 2. 選出特定股票
stock_df = df[df['股票代碼'] == stock_id].dropna(subset=features + [target])
if len(stock_df) <= sequence_length:
    raise Exception(f"[ERROR] Stock {stock_id}: Not enough data.")

# 3. 特徵scaling（*必須和訓練時一樣*）
feature_data = stock_df[features].values
target_data = stock_df[[target]].values
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
scaled_features = feature_scaler.fit_transform(feature_data)
scaled_target = target_scaler.fit_transform(target_data)

# 4. 組出最近一筆可預測的sequence
X_pred = scaled_features[-sequence_length:]  # shape: (sequence_length, num_features)
X_pred = np.expand_dims(X_pred, axis=0)     # shape: (1, sequence_length, num_features)

# 5. 載入模型並預測
model = keras.models.load_model(model_path)
y_pred_scaled = model.predict(X_pred)
y_pred = target_scaler.inverse_transform(y_pred_scaled)[0][0]

print(f"[PREDICT] 股票代碼 {stock_id}，下一天預測收盤價：{y_pred:.2f}")