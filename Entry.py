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

# Hàm tính nến Heikin Ashi
def calculate_heikin_ashi(data):
    data['HA_Close'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    data['HA_Open'] = (data['open'].shift(1) + data['close'].shift(1)) / 2
    data['HA_High'] = data[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    data['HA_Low'] = data[['low', 'HA_Open', 'HA_Close']].min(axis=1)
    return data

# Hàm tính RSI
def calculate_rsi(data, window):
    delta = data['HA_Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Hàm tính MACD
def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data['HA_Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['HA_Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Hàm lấy dữ liệu thời gian thực từ Binance
def get_realtime_klines(symbol, interval, lookback, client, end_time=None):
    if end_time:
        klines = client.futures_klines(symbol=symbol, interval=interval, endTime=int(end_time.timestamp() * 1000), limit=lookback)
    else:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    data[['open', 'high', 'low', 'close']] = data[['open', 'high', 'low', 'close']].astype(float)
    data['volume'] = data['volume'].astype(float)
    
    # Tính toán nến Heikin Ashi
    data = calculate_heikin_ashi(data)
    return data

# Hàm phân tích xu hướng cho từng khung thời gian
def analyze_trend(interval, client):
    symbol = 'BTCUSDT'
    lookback = 1000
    data = get_realtime_klines(symbol, interval, lookback, client)
    
    # Sử dụng giá Heikin Ashi
    rsi = calculate_rsi(data, 14)
    macd, signal_line = calculate_macd(data)

    data['target'] = (data['HA_Close'].shift(-1) > data['HA_Close']).astype(int)
    data['rsi'] = rsi
    data['macd'] = macd
    data['signal_line'] = signal_line
    
    features = data[['rsi', 'macd', 'signal_line']].dropna()
    target = data['target'].dropna()

    min_length = min(len(features), len(target))
    features = features.iloc[:min_length]
    target = target.iloc[:min_length]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, target

# Hàm dự báo xu hướng và trả về chuỗi kết quả
def get_trend():
    api_key = 'YOUR_API_KEY'
    api_secret = 'YOUR_API_SECRET'
    client = Client(api_key, api_secret, tld='com', testnet=False)
    
    # Phân tích xu hướng cho các khung thời gian M1, M15, H4
    X_m1, y_m1 = analyze_trend(Client.KLINE_INTERVAL_1MINUTE, client)
    X_m15, y_m15 = analyze_trend(Client.KLINE_INTERVAL_15MINUTE, client)
    X_h4, y_h4 = analyze_trend(Client.KLINE_INTERVAL_4HOUR, client)

    # Đồng bộ kích thước của các tập dữ liệu
    min_len = min(len(X_m1), len(X_m15), len(X_h4))
    X_m1, y_m1 = X_m1[-min_len:], y_m1[-min_len:]
    X_m15, y_m15 = X_m15[-min_len:], y_m15[-min_len:]
    X_h4, y_h4 = X_h4[-min_len:], y_h4[-min_len:]

    # Kết hợp dữ liệu từ các mô hình khác nhau
    X_combined = np.hstack([X_m1, X_m15, X_h4])
    y_combined = y_m1  # Giả sử tất cả các y đều giống nhau về kích thước và đồng nhất về kết quả

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    # Định nghĩa các mô hình cơ bản
    base_models = [
        ('m1_model', LogisticRegression(max_iter=1000)),
        ('m15_model', LogisticRegression(max_iter=1000)),
        ('h4_model', LogisticRegression(max_iter=1000))
    ]

    # Mô hình meta (Logistic Regression)
    meta_model = LogisticRegression(max_iter=1000)

    # Xây dựng mô hình stacking
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    stacking_model.fit(X_train, y_train)

    # Dự đoán xu hướng giá thời gian thực
    latest_features = X_combined[-1].reshape(1, -1)
    prediction_prob = stacking_model.predict_proba(latest_features)[0]

    # Ngưỡng cho xu hướng không rõ ràng
    threshold = 0.45

    if prediction_prob[1] > 1 - threshold:
        trend = "tăng"
    elif prediction_prob[1] < threshold:
        trend = "giảm"
    else:
        trend = "xu hướng không rõ ràng"

    # Trả về xu hướng
    return trend
