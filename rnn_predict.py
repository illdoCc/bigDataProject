import os
import numpy as np
import pandas as pd
import joblib

def predict_price(stock_id, stock_df, model_dir='./models/rnn', sequence_length=7, features=None):
    """
    !!!此為單筆股票預測
    根據股票代碼與最近7天資料預測下一天的股價。
    
    參數:
        stock_id (str or int): 股票代碼
        stock_df (pd.DataFrame): 包含該股票的歷史資料（必須包含指定的 features 欄位）
        model_dir (str): 模型儲存資料夾
        sequence_length (int): 每次輸入模型的時間步長（預設為 7）
        features (list): 特徵欄位清單（要與訓練時一致）
        
    回傳:
        float: 預測的下一天股價（已反正規化）
    """
    if features is None:
        features = ['最後揭示買價', '最高價', '高低價差', '最後揭示賣價', '本益比_TEJ',
                    '最低價', '開盤價', '股價淨值比_TEJ', '股利殖利率', '成交金額比重', '周轉率']
    
    # 載入模型與scalers
    pkl_path = os.path.join(model_dir, f'{stock_id}_RNN_model.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Model file for stock {stock_id} not found: {pkl_path}")
    
    data = joblib.load(pkl_path)
    model = data['model']
    feature_scaler = data['feature_scaler']
    target_scaler = data['target_scaler']
    
    # 檢查資料是否夠長
    stock_df = stock_df.sort_values(by='日期')
    stock_df = stock_df.dropna(subset=features)
    if len(stock_df) < sequence_length:
        raise ValueError(f"Not enough data to make prediction for stock {stock_id}")
    
    # 擷取最近 sequence_length 筆資料作為輸入
    recent_data = stock_df[features].values[-sequence_length:]
    scaled_input = feature_scaler.transform(recent_data)
    X_input = np.expand_dims(scaled_input, axis=0)  # 變成 shape: (1, 7, num_features)
    
    # 預測 & 反正規化
    y_pred_scaled = model.predict(X_input, verbose=0)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)[0][0]
    
    return y_pred