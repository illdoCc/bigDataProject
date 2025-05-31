import pandas as pd
import numpy as np
import joblib
import json
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import OneHotEncoder

# === 參數 ===
MODEL_DIR = 'models/arima'
INPUT_DATA_PATH = 'price.csv'
OUTPUT_PATH = 'predictions.json'
FORECAST_DAYS = 31  # 包含 0~30 天

# === 日期對照表（你可以從 json 載入，也可直接貼）===
date_map = {
    "0": "2022/8/29",
    "1": "2022/8/30",
    "2": "2022/8/31",
    "3": "2022/9/1",
    "4": "2022/9/2",
    "5": "2022/9/5",
    "6": "2022/9/6",
    "7": "2022/9/7",
    "8": "2022/9/8",
    "9": "2022/9/12",
    "10": "2022/9/13",
    "11": "2022/9/14",
    "12": "2022/9/15",
    "13": "2022/9/16",
    "14": "2022/9/19",
    "15": "2022/9/20",
    "16": "2022/9/21",
    "17": "2022/9/22",
    "18": "2022/9/23",
    "19": "2022/9/26",
    "20": "2022/9/27",
    "21": "2022/9/28",
    "22": "2022/9/29",
    "23": "2022/9/30",
    "24": "2022/10/3",
    "25": "2022/10/4",
    "26": "2022/10/5",
    "27": "2022/10/6",
    "28": "2022/10/7",
    "29": "2022/10/11",
    "30": "2022/10/12"
}
# 注意：如果要 0~31（共32天），最後再加上 "31": "2022/10/13"
# 如果不需要，這個設定已經是 0~30 共31天

# === 讀取資料 ===
try:
    data = pd.read_csv(INPUT_DATA_PATH, encoding='big5')
    data['日期'] = pd.to_datetime(data['日期'])
    data.sort_values(['股票代碼', '日期'], inplace=True)
except Exception as e:
    print(f"❌ 讀取輸入資料失敗: {e}")
    exit()

required_columns = [
    '類別', '股票代碼', '日期', '最後揭示買價', '最低價', '本益比_TEJ', '最高價',
    '收盤價', '開盤價', '周轉率', '報酬率', '股價淨值比_TEJ', '成交金額比重',
    '最後揭示賣價', '股利殖利率', '高低價差'
]
if not all(col in data.columns for col in required_columns):
    print(f"❌ 輸入資料缺少必要欄位，需包含: {required_columns}")
    exit()

# OneHot 編碼（與訓練時一致）
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_df = pd.DataFrame(
    ohe.fit_transform(data[['類別']]),
    columns=ohe.get_feature_names_out(['類別']),
    index=data.index
)
data = pd.concat([data.drop(columns=['類別']), ohe_df], axis=1)

# 預測結果初始化（key 為天數字串）
predictions = {i: {} for i in range(FORECAST_DAYS)}

# 主要預測迴圈
for stock_id, group in data.groupby('股票代碼'):
    print(f"✅ 處理股票: {stock_id}")
    group = group.set_index('日期')
    X = group.drop(columns=['收盤價', '股票代碼'])
    y = group['收盤價']

    # 載入預訓練模型
    model_path = os.path.join(MODEL_DIR, f'{stock_id}.pkl')
    try:
        model_dict = joblib.load(model_path)
        model_fit = model_dict['model']
        order = model_dict['order']
    except Exception as e:
        print(f"❌ 找不到模型或載入失敗: {stock_id}, 錯誤: {e}")
        continue

    # 依照天數 key 預測（用你給的日期 json）
    for day in range(FORECAST_DAYS):
        date_str = date_map[str(day)]
        # 直接轉 datetime，且強制格式一致
        current_date = pd.to_datetime(date_str, format="%Y/%m/%d")
        # 這裡用 .get_loc 仍可能有 nanosecond-level 差異，用 == 配合 numpy where 最穩
        date_mask = (group.index == current_date)
        if not date_mask.any():
            print(f"股票 {stock_id} 缺少 {date_str} 的資料，跳過")
            continue
        idx = np.where(date_mask)[0][0]

        X_train = X.iloc[:idx+1]
        y_train = y.iloc[:idx+1]
        X_current = X.iloc[idx:idx+1]

        if y_train.isna().any():
            print(f"❌ 股票 {stock_id} 在 {date_str} 訓練資料含 NaN，跳過")
            continue
        if not (X_train.columns == X_current.columns).all():
            print(f"❌ 股票 {stock_id} 在 {date_str} 特徵欄位不一致，跳過")
            continue

        try:
            model_updated = SARIMAX(
                y_train, exog=X_train,
                order=order, enforce_stationarity=False, enforce_invertibility=False
            ).fit(disp=False)
            forecast = model_updated.forecast(steps=1, exog=X_current)
            predicted_price = forecast.iloc[0]
            predictions[day][int(stock_id)] = float(predicted_price)
            print(f"✅ 股票: {stock_id}，{date_str} 預測收盤價: {predicted_price:.2f}")
        except Exception as e:
            print(f"❌ 股票 {stock_id} 在 {date_str} 預測失敗: {e}")
            continue

# 輸出預測結果到 txt，格式即為你要的格式
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(json.dumps(predictions, ensure_ascii=False, indent=4))
print(f"✅ 預測結果已儲存至: {OUTPUT_PATH}")
