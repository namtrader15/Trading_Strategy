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

# Các hàm tính RSI và MACD vẫn giữ nguyên
print(f"NAMTRADER_TRENDLINEBTC_FUTURES")

def calculate_rsi(data, window):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data['close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Hàm lấy dữ liệu từ Binance
def get_realtime_klines(symbol, interval, lookback, client, end_time=None):
    if end_time:
        klines = client.futures_klines(symbol=symbol, interval=interval, endTime=int(end_time.timestamp() * 1000), limit=lookback)
    else:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    data[['open', 'high', 'low', 'close']] = data[['open', 'high', 'low', 'close']].astype(float)
    data['volume'] = data['volume'].astype(float)
    return data

# Hàm phân tích xu hướng cho từng khung thời gian
def analyze_trend(interval, client):
    symbol = 'BTCUSDT'
    lookback = 1000
    data = get_realtime_klines(symbol, interval, lookback, client)
    rsi = calculate_rsi(data, 14)
    macd, signal_line = calculate_macd(data)

    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
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
    
    # Phân tích xu hướng cho các khung thời gian M5, H1, D1
    X_m5, y_m5 = analyze_trend(Client.KLINE_INTERVAL_5MINUTE, client)
    X_h1, y_h1 = analyze_trend(Client.KLINE_INTERVAL_1HOUR, client)
    X_d1, y_d1 = analyze_trend(Client.KLINE_INTERVAL_1DAY, client)

    # Đồng bộ kích thước của các tập dữ liệu
    min_len = min(len(X_m5), len(X_h1), len(X_d1))
    X_m5, y_m5 = X_m5[-min_len:], y_m5[-min_len:]
    X_h1, y_h1 = X_h1[-min_len:], y_h1[-min_len:]
    X_d1, y_d1 = X_d1[-min_len:], y_d1[-min_len:]

    # Kết hợp dữ liệu từ các mô hình khác nhau
    X_combined = np.hstack([X_m5, X_h1, X_d1])
    y_combined = y_m5  # Giả sử tất cả các y đều giống nhau về kích thước và đồng nhất về kết quả

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    # Định nghĩa các mô hình cơ bản
    base_models = [
        ('m5_model', LogisticRegression(max_iter=1000)),
        ('h1_model', LogisticRegression(max_iter=1000)),
        ('d1_model', LogisticRegression(max_iter=1000))
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
