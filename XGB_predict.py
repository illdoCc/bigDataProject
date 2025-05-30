# import os

# # ====== 自訂路徑 ======
# base_dir = "/content/drive/MyDrive/Arima_XGB_RF/XGBoost"
# input_path = "/content/drive/MyDrive/price_labeled.csv"
# txt_dir = f"{base_dir}/txt"
# png_dir = f"{base_dir}/png"
# model_dir = f"{base_dir}/models"

# # 確保輸出資料夾存在
# os.makedirs(base_dir, exist_ok=True)
# os.makedirs(txt_dir, exist_ok=True)
# os.makedirs(png_dir, exist_ok=True)
# os.makedirs(model_dir, exist_ok=True)
# print(f"確保資料夾存在: {base_dir}, {txt_dir}, {png_dir}, {model_dir}")

# # ====== 匯入套件 ======
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import xgboost as xgb
# from joblib import dump
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

# # ====== 定義 MAPE 函數 ======
# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     epsilon = 1e-8
#     return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# # ====== 讀取資料與前處理 ======
# try:
#     df = pd.read_csv(input_path, encoding='big5')
#     print("資料讀取成功")
#     print("資料欄位:", df.columns.tolist())
# except Exception as e:
#     print(f"資料讀取失敗: {str(e)}")
#     raise

# # 檢查類別欄位是否存在
# category_col = '類別'
# if category_col not in df.columns:
#     possible_cols = [col for col in df.columns if '類別' in col or 'category' in col.lower()]
#     if possible_cols:
#         category_col = possible_cols[0]
#         print(f"未找到 '類別' 欄位，使用 '{category_col}' 代替")
#     else:
#         print("錯誤: 資料中未找到 '類別' 或類似欄位")
#         raise KeyError("類別欄位未找到")

# df = df.sort_values(['股票代碼', '日期'])
# df['next_close'] = df.groupby('股票代碼')['收盤價'].shift(-1)
# df = df.dropna(subset=['next_close']).reset_index(drop=True)
# df['stock_code'] = df['股票代碼'].astype(str)
# df['category'] = df[category_col]
# df = pd.concat([df, pd.get_dummies(df[category_col], prefix='cat')], axis=1)
# df.drop(columns=[category_col], inplace=True)
# df = df.sort_values(by=['日期', 'stock_code']).reset_index(drop=True)

# # ====== 檢查每支股票的資料量 ======
# stocks = df['stock_code'].unique()
# print(f"總股票數: {len(stocks)}")
# for stock in stocks:
#     stock_data = df[df['stock_code'] == stock]
#     print(f"股票 {stock} 資料筆數: {len(stock_data)}")

# # ====== 檢查本益比缺失 ======
# stocks_without_pe = []
# for stock in stocks:
#     stock_data = df[df['stock_code'] == stock]
#     if stock_data['本益比_TEJ'].isna().any():
#         stocks_without_pe.append(stock)
# print(f"本益比缺失的股票: {stocks_without_pe}")

# # ====== 按股票訓練模型並計算 MSE/MAPE ======
# stock_metrics = []
# categories = df['category'].unique()
# for stock in stocks:
#     print(f"訓練股票 {stock} 的 XGBoost 模型")
#     stock_data = df[df['stock_code'] == stock].reset_index(drop=True)  # 重置索引
    
#     # 確認資料量
#     N = len(stock_data)
#     if N < 30:
#         print(f"股票 {stock} 資料不足 (少於 30 筆)，跳過")
#         continue
    
#     # 按 70:15:15 時序分割資料
#     train_end = int(N * 0.7)
#     val_end = train_end + int(N * 0.15)
#     df_train = stock_data.iloc[:train_end].reset_index(drop=True)
#     df_val = stock_data.iloc[train_end:val_end].reset_index(drop=True)
#     df_test = stock_data.iloc[val_end:].reset_index(drop=True)
    
#     print(f"股票 {stock} 分割: 訓練 {len(df_train)} 筆 ({df_train['日期'].iloc[0]} 到 {df_train['日期'].iloc[-1]}), "
#           f"驗證 {len(df_val)} 筆 ({df_val['日期'].iloc[0]} 到 {df_val['日期'].iloc[-1]}), "
#           f"測試 {len(df_test)} 筆 ({df_test['日期'].iloc[0]} 到 {df_test['日期'].iloc[-1]})")
    
#     if len(df_train) < 20 or len(df_val) < 1 or len(df_test) < 1:
#         print(f"股票 {stock} 分割後資料不足，跳過")
#         continue
    
#     # 特徵處理與標準化
#     feature_cols = ['收盤價', '開盤價', '最高價', '最低價', '報酬率']  # 優化特徵，僅選取與價格相關的特徵
#     X_train = df_train[feature_cols]
#     X_val = df_val[feature_cols]
#     X_test = df_test[feature_cols]
#     y_train = df_train['next_close'].values
#     y_val = df_val['next_close'].values
#     y_test = df_test['next_close'].values
#     numeric_features = feature_cols
    
#     try:
#         scaler = StandardScaler().fit(X_train[numeric_features])
#         X_train[numeric_features] = scaler.transform(X_train[numeric_features])
#         X_val[numeric_features] = scaler.transform(X_val[numeric_features])
#         X_test[numeric_features] = scaler.transform(X_test[numeric_features])
#     except Exception as e:
#         print(f"股票 {stock} 標準化失敗: {str(e)}")
#         continue
    
#     # 訓練模型
#     xgb_model = xgb.XGBRegressor(
#         objective='reg:squarederror',
#         learning_rate=0.1,
#         n_estimators=100,  # 減少複雜度
#         max_depth=3,       # 減少深度
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42
#     )
#     try:
#         xgb_model.fit(X_train, y_train)
#         print(f"股票 {stock} 模型訓練成功")
#     except Exception as e:
#         print(f"股票 {stock} 模型訓練失敗: {str(e)}")
#         continue
    
#     # 儲存模型
#     model_filename = f"{model_dir}/xgb_model_{stock}.pkl"
#     try:
#         dump(xgb_model, model_filename)
#         if os.path.exists(model_filename):
#             print(f"模型已儲存至: {model_filename}")
#         else:
#             print(f"模型儲存失敗: {model_filename}")
#     except Exception as e:
#         print(f"股票 {stock} 模型儲存失敗: {str(e)}")
#         continue
    
#     # 預測與 MSE/MAPE 評估
#     try:
#         y_val_pred = xgb_model.predict(X_val)
#         val_mse = mean_squared_error(y_val, y_val_pred)
#         val_mape = mean_absolute_percentage_error(y_val, y_val_pred)
        
#         y_test_pred = xgb_model.predict(X_test)
#         test_mse = mean_squared_error(y_test, y_test_pred)
#         test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
        
#         # 診斷：輸出部分預測與實際值
#         print(f"股票 {stock} 預測診斷 - 驗證集首5筆: 實際 {y_val[:5]}, 預測 {y_val_pred[:5]}")
#         print(f"股票 {stock} 預測診斷 - 測試集首5筆: 實際 {y_test[:5]}, 預測 {y_test_pred[:5]}")
        
#         # 記錄股票的類別與指標
#         category = stock_data['category'].iloc[0]
#         stock_metrics.append({
#             'stock_code': stock,
#             'category': category,
#             'val_mse': val_mse,
#             'val_mape': val_mape,
#             'test_mse': test_mse,
#             'test_mape': test_mape
#         })
#     except Exception as e:
#         print(f"股票 {stock} 預測或評估失敗: {str(e)}")
#         continue


import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load

def predict_next_day_with_xgb(input_df, model_dir, feature_cols):
    """
    使用儲存的 XGBoost 模型預測每檔股票下一筆收盤價（僅用最新一筆資料）。
    
    Parameters:
        input_df (pd.DataFrame): 包含所有股票的資料（必須有 '股票代碼' 與必要特徵）。
        model_dir (str): 權重檔目錄，例如 "models/xgb"
        feature_cols (list): 預測用的特徵欄位
        
    Returns:
        dict: 每檔股票預測的下一筆收盤價
    """
    from joblib import load
    from sklearn.preprocessing import StandardScaler
    import os

    predictions = {}

    if '股票代碼' not in input_df.columns:
        raise ValueError("輸入資料必須包含 '股票代碼' 欄位")

    input_df = input_df.sort_values(['股票代碼', '日期'])
    input_df['stock_code'] = input_df['股票代碼'].astype(str)
    stocks = input_df['stock_code'].unique()

    for stock in stocks:
        model_path = os.path.join(model_dir, f"xgb_model_{stock}.pkl")
        if not os.path.exists(model_path):
            print(f"❌ 模型不存在：{stock}")
            continue

        stock_data = input_df[input_df['stock_code'] == stock].copy()

        if len(stock_data) < 1:
            print(f"⚠️ 股票 {stock} 沒有資料，跳過")
            continue

        latest_row = stock_data.tail(1)

        if latest_row[feature_cols].isnull().any().any():
            print(f"⚠️ 股票 {stock} 最新一筆資料有缺失值，跳過")
            continue

        try:
            scaler = StandardScaler().fit(stock_data[feature_cols])  # 模擬訓練階段使用的標準化
            X_latest = scaler.transform(latest_row[feature_cols])
            model = load(model_path)
            y_pred = model.predict(X_latest)
            predictions[stock] = y_pred[0]
            print(f"✅ {stock} 預測成功：{y_pred[0]}")
        except Exception as e:
            print(f"❌ {stock} 預測失敗: {e}")
    
    return predictions