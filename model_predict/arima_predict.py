import pandas as pd
import numpy as np
import os
import joblib
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder

# ✅ 設定路徑
DATA_PATH = '/content/drive/MyDrive/price.csv'  # 修改為你的資料路徑
SAVE_DIR = '/content/drive/MyDrive/Arima_XGB_RF/Arima'
MODEL_DIR = os.path.join(SAVE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ 讀取資料
raw_df = pd.read_csv(DATA_PATH, encoding='big5')
raw_df['日期'] = pd.to_datetime(raw_df['日期'])
raw_df.sort_values(['股票代碼', '日期'], inplace=True)

# ✅ 建立 OneHot 編碼器 (for 類別)
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_df = pd.DataFrame(
    ohe.fit_transform(raw_df[['類別']]),
    columns=ohe.get_feature_names_out(['類別']),
    index=raw_df.index
)
df = pd.concat([raw_df.drop(columns=['類別']), ohe_df], axis=1)

# ✅ 平移收盤價作為目標變數（隔日收盤）
df['target'] = df.groupby('股票代碼')['收盤價'].shift(-1)
df.dropna(inplace=True)

# ✅ 以產業為單位記錄報告與圖片資料
industry_reports = {}
industry_forecasts = {}

# ✅ 開始建模
for stock_id, group in df.groupby('股票代碼'):
    industry = raw_df[raw_df['股票代碼'] == stock_id]['類別'].iloc[0]
    group = group.set_index('日期')

    # 目標值與特徵欄位
    y = group['target']
    X = group.drop(columns=['target', '收盤價', '股票代碼'])

    if len(group) < 100:
        continue  # 資料太少跳過

    # 時序切分
    total_len = len(group)
    train_end = int(total_len * 0.8)
    valid_end = int(total_len * 0.9)

    y_train = y[:train_end]
    y_valid = y[train_end:valid_end]
    y_test = y[valid_end:]

    X_train = X[:train_end]
    X_valid = X[train_end:valid_end]
    X_test = X[valid_end:]

    # 自動 ARIMAX
    try:
        model_auto = auto_arima(
            y_train, exogenous=X_train,
            start_p=1, start_q=1, max_p=4, max_q=4,
            seasonal=False, stepwise=True, suppress_warnings=True
        )
        order = model_auto.order
    except:
        continue

    # 訓練最終模型（訓練+驗證）
    y_tv = pd.concat([y_train, y_valid])
    X_tv = pd.concat([X_train, X_valid])
    model = SARIMAX(y_tv, exog=X_tv, order=order,
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    # 預測
    forecast = model_fit.forecast(steps=len(y_test), exog=X_test)
    mape = mean_absolute_percentage_error(y_test, forecast)

    # 儲存模型
    joblib.dump({'model': model_fit, 'order': order}, os.path.join(MODEL_DIR, f'{stock_id}.pkl'))

    # 建立產業分組記錄
    if industry not in industry_reports:
        industry_reports[industry] = []
        industry_forecasts[industry] = []

    industry_reports[industry].append(f"股票: {stock_id}, Order: {order}, MAPE: {mape:.4f}")
    industry_forecasts[industry].append((stock_id, y_test.index, y_test.values, forecast))















# ✅ 匯入套件
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

# ✅ 自訂參數
MODEL_PATH = '/content/drive/MyDrive/your_folder/output/models/{stock_id}.pkl'  # 預訓練模型位置
NEW_DATA_PATH = '/content/drive/MyDrive/your_folder/daily_update.csv'  # 每日更新資料位置（包含今天特徵與昨天實際收盤價）
UPDATED_MODEL_PATH = '/content/drive/MyDrive/your_folder/output/models/{stock_id}_updated.pkl'

# ✅ 讀取更新資料（格式與訓練資料一致）
data = pd.read_csv(NEW_DATA_PATH, encoding='big5')
data['日期'] = pd.to_datetime(data['日期'])
data.sort_values(['股票代碼', '日期'], inplace=True)

# ✅ 載入模型並微調、預測
predictions = []
for stock_id, group in data.groupby('股票代碼'):
    if len(group) < 2:
        continue  # 至少要有昨天與今天資料

    # 準備資料
    group = group.set_index('日期')
    X = group.drop(columns=['收盤價', '股票代碼'])
    y = group['收盤價']

    X_train = X.iloc[:-1]  # 昨天的特徵
    y_train = y.iloc[:-1]  # 昨天的實際收盤
    X_today = X.iloc[-1:]  # 今天的特徵

    # 載入模型與參數
    try:
        model_dict = joblib.load(MODEL_PATH.format(stock_id=stock_id))
        model_fit = model_dict['model']
        order = model_dict['order']
    except:
        print(f"❌ 找不到模型: {stock_id}")
        continue

    # 更新模型（用昨天資料微調）
    model_updated = SARIMAX(
        y_train, exog=X_train,
        order=order, enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)

    # 儲存更新後模型（選擇性）
    joblib.dump({'model': model_updated, 'order': order}, UPDATED_MODEL_PATH.format(stock_id=stock_id))

    # 用今天資料預測
    forecast = model_updated.forecast(steps=1, exog=X_today)
    predictions.append((stock_id, forecast.iloc[0]))

# ✅ 顯示預測結果
for stock_id, pred in predictions:
    print(f"股票: {stock_id}，預測今日收盤價: {pred:.2f}")
 