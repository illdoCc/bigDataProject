import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout

#Read all data
df = pd.read_csv('price_new.csv', encoding='big5')
df['日期'] = pd.to_datetime(df['日期'])
df.sort_values(by=['股票代碼', '日期'], inplace=True)

features = ['最後揭示買價', '最高價', '高低價差', '最後揭示賣價', '本益比_TEJ',
            '最低價', '開盤價', '股價淨值比_TEJ', '股利殖利率', '成交金額比重', '周轉率']
target = '收盤價'
sequence_length = 30  # 改為吃30天資料

#Output directory
output_dir = './models_per_stock_lstm'
os.makedirs(output_dir, exist_ok=True)

#Train for each stock
stock_ids = df['股票代碼'].unique()

for stock_id in stock_ids:
    group = df[df['股票代碼'] == stock_id].dropna(subset=features + [target])
    if len(group) <= sequence_length:
        print(f"[SKIP] Stock {stock_id}: Not enough data.")
        continue

    feature_data = group[features].values
    target_data = group[[target]].values

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(feature_data)
    scaled_target = target_scaler.fit_transform(target_data)

    X, y = [], []
    for i in range(len(group) - sequence_length):
        X.append(scaled_features[i:i+sequence_length])
        y.append(scaled_target[i + sequence_length][0])

    X = np.array(X)
    y = np.array(y)

    if len(X) < 10:
        print(f"[SKIP] Stock {stock_id}: Too few sequences.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    #Build LSTM model
    model = keras.models.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, len(features))),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    #Train model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                        validation_data=(X_test, y_test), verbose=0)

    #Save as pkl file
    pkl_path = os.path.join(output_dir, f'{stock_id}_LSTM_model.pkl')
    joblib.dump({
        'model': model,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }, pkl_path)

    # === Save training loss plot ===
    loss_plot_path = os.path.join(output_dir, f'{stock_id}_loss_plot.png')
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training Loss - Stock {stock_id} (LSTM)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

    #Predict and show first 5 results
    y_pred_scaled = model.predict(X_test)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    print(f"\n[RESULT] Stock {stock_id} LSTM prediction (first 5):")
    for i in range(min(5, len(y_pred))):
        pred = y_pred[i][0]
        real = y_true[i][0]
        error = abs(pred - real)
        print(f"{i+1}: Predicted = {pred:.2f}, Actual = {real:.2f}, Error = {error:.2f}")

    print(f"[OK] {stock_id}: LSTM model, scalers, predictions, and loss plot saved.\n")