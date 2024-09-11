from binance.client import Client
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import StackingClassifier
import numpy as np
from datetime import datetime
import time

# Các hàm tính RSI và MACD
def calculate_rsi(data, window):
    delta = data['HA_Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data['HA_Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['HA_Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Hàm tính toán Heikin Ashi từ dữ liệu nến thông thường
def convert_to_heikin_ashi(data):
    ha_data = data.copy()
    ha_data['HA_Close'] = (ha_data['open'] + ha_data['high'] + ha_data['low'] + ha_data['close']) / 4
    ha_data['HA_Open'] = (ha_data['open'].shift(1) + ha_data['close'].shift(1)) / 2
    ha_data['HA_High'] = ha_data[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    ha_data['HA_Low'] = ha_data[['low', 'HA_Open', 'HA_Close']].min(axis=1)
    ha_data = ha_data.dropna()
    return ha_data[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]

# Hàm lấy dữ liệu từ Binance (real-time)
def get_realtime_klines(symbol, interval, lookback, client):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    data[['open', 'high', 'low', 'close']] = data[['open', 'high', 'low', 'close']].astype(float)
    return data

# Hàm phân tích xu hướng với Heikin Ashi cho từng khung thời gian
def analyze_trend_with_heikin(interval, client):
    symbol = 'BTCUSDT'
    lookback = 1000
    data = get_realtime_klines(symbol, interval, lookback, client)
    ha_data = convert_to_heikin_ashi(data)
    rsi = calculate_rsi(ha_data, 14)
    macd, signal_line = calculate_macd(ha_data)
    ha_data['target'] = (ha_data['HA_Close'].shift(-1) > ha_data['HA_Close']).astype(int)
    ha_data['rsi'] = rsi
    ha_data['macd'] = macd
    ha_data['signal_line'] = signal_line
    features = ha_data[['rsi', 'macd', 'signal_line']].dropna()
    target = ha_data['target'].dropna()
    min_length = min(len(features), len(target))
    features = features.iloc[:min_length]
    target = target.iloc[:min_length]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, target

# Hàm dự đoán xu hướng và trả về kết quả xu hướng
def get_trend():
    api_key = 'YOUR_API_KEY'
    api_secret = 'YOUR_API_SECRET'
    client = Client(api_key, api_secret, tld='com', testnet=False)
    X_h4, y_h4 = analyze_trend_with_heikin(Client.KLINE_INTERVAL_4HOUR, client)
    X_d1, y_d1 = analyze_trend_with_heikin(Client.KLINE_INTERVAL_1DAY, client)
    X_w1, y_w1 = analyze_trend_with_heikin(Client.KLINE_INTERVAL_1WEEK, client)
    min_len = min(len(X_h4), len(X_d1), len(X_w1))
    X_combined = np.hstack([X_h4[-min_len:], X_d1[-min_len:], X_w1[-min_len:]])
    y_combined = y_h4[-min_len:]
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
    base_models = [
        ('h4_model', LogisticRegression(max_iter=1000)),
        ('d1_model', LogisticRegression(max_iter=1000)),
        ('w1_model', LogisticRegression(max_iter=1000))
    ]
    meta_model = LogisticRegression(max_iter=1000)
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    stacking_model.fit(X_train, y_train)
    latest_features = X_combined[-1].reshape(1, -1)
    prediction_prob = stacking_model.predict_proba(latest_features)[0]
    threshold = 0.45
    if prediction_prob[1] > 1 - threshold:
        trend = "tăng"
        
    elif prediction_prob[1] < threshold:
        trend = "giảm"
        
    else:
        trend = "xu hướng không rõ ràng"
        
    return trend
    
