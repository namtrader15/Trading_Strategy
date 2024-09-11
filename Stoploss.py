from binance.client import Client
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import StackingClassifier  # Thêm dòng này để sửa lỗi
import numpy as np
from datetime import datetime
import time
from playsound import playsound
print(f"NAMTRADER_SWING_TRADER")
# Các hàm tính RSI và MACD vẫn giữ nguyên
def calculate_rsi(data, window):
    delta = data['HA_Close'].diff()  # Sửa lại để sử dụng 'HA_Close'
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data['HA_Close'].ewm(span=fast, adjust=False).mean()  # Sửa lại để sử dụng 'HA_Close'
    exp2 = data['HA_Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Hàm tính toán Heikin Ashi từ dữ liệu nến thông thường
def convert_to_heikin_ashi(data):
    ha_data = data.copy()

    # Tính toán Heikin Ashi
    ha_data['HA_Close'] = (ha_data['open'] + ha_data['high'] + ha_data['low'] + ha_data['close']) / 4
    ha_data['HA_Open'] = (ha_data['open'].shift(1) + ha_data['close'].shift(1)) / 2
    ha_data['HA_High'] = ha_data[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    ha_data['HA_Low'] = ha_data[['low', 'HA_Open', 'HA_Close']].min(axis=1)

    # Loại bỏ các giá trị NaN do shift
    ha_data = ha_data.dropna()
    
    return ha_data[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]

# Hàm lấy dữ liệu từ Binance (real-time)
def get_realtime_klines(symbol, interval, lookback, client):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    data[['open', 'high', 'low', 'close']] = data[['open', 'high', 'low', 'close']].astype(float)
    data['volume'] = data['volume'].astype(float)
    return data

# Hàm phân tích xu hướng với Heikin Ashi cho từng khung thời gian
def analyze_trend_with_heikin(interval, client):
    symbol = 'BTCUSDT'
    lookback = 1000
    # Lấy dữ liệu giá real-time
    data = get_realtime_klines(symbol, interval, lookback, client)

    # Chuyển đổi sang nến Heikin Ashi
    ha_data = convert_to_heikin_ashi(data)

    # Tính RSI và MACD sử dụng dữ liệu Heikin Ashi
    rsi = calculate_rsi(ha_data, 14)
    macd, signal_line = calculate_macd(ha_data)

    ha_data['target'] = (ha_data['HA_Close'].shift(-1) > ha_data['HA_Close']).astype(int)  # Sử dụng 'HA_Close'
    ha_data['rsi'] = rsi
    ha_data['macd'] = macd
    ha_data['signal_line'] = signal_line

    # Loại bỏ các giá trị NaN
    features = ha_data[['rsi', 'macd', 'signal_line']].dropna()
    target = ha_data['target'].dropna()

    min_length = min(len(features), len(target))
    features = features.iloc[:min_length]
    target = target.iloc[:min_length]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target

# Hàm meta-stacking với Heikin Ashi
def meta_stacking_with_heikin_ashi():
    api_key = 'YOUR_API_KEY'
    api_secret = 'YOUR_API_SECRET'
    client = Client(api_key, api_secret, tld='com', testnet=False)

    # Phân tích xu hướng với Heikin Ashi cho các khung thời gian H1, H4, D1
    X_h1, y_h1 = analyze_trend_with_heikin(Client.KLINE_INTERVAL_1HOUR, client)
    X_h4, y_h4 = analyze_trend_with_heikin(Client.KLINE_INTERVAL_4HOUR, client)
    X_d1, y_d1 = analyze_trend_with_heikin(Client.KLINE_INTERVAL_1DAY, client)

    # Đồng bộ kích thước của các tập dữ liệu
    min_len = min(len(X_h1), len(X_h4), len(X_d1))
    X_combined = np.hstack([X_h1[-min_len:], X_h4[-min_len:], X_d1[-min_len:]])
    y_combined = y_h1[-min_len:]  # Giả sử nhãn giống nhau

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    # Định nghĩa các mô hình con và mô hình meta (Logistic Regression)
    base_models = [
        ('h1_model', LogisticRegression(max_iter=1000)),
        ('h4_model', LogisticRegression(max_iter=1000)),
        ('d1_model', LogisticRegression(max_iter=1000))
    ]

    meta_model = LogisticRegression(max_iter=1000)

    # Xây dựng mô hình stacking
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    stacking_model.fit(X_train, y_train)

    # Đánh giá mô hình
    y_pred = stacking_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Dự đoán xu hướng thời gian thực
    latest_features = X_combined[-1].reshape(1, -1)
    prediction_prob = stacking_model.predict_proba(latest_features)[0]
    prediction = stacking_model.predict(latest_features)
    #Price
    btc_price = client.get_symbol_ticker(symbol="BTCUSDT")
    # Ngưỡng dự đoán
    threshold = 0.45
    if prediction_prob[1] > 1 - threshold:
        trend = "Tăng"
        print(f" - Lệnh đang chạy an toàn...")
    elif prediction_prob[1] < threshold:
        trend = "Giảm"
        print(f" - Lệnh đang chạy an toàn... ")
    else:
        trend = " Không rõ ràng "
        print(f"StopLoss: {btc_price['price']} USDT")
        playsound(r"C:\Users\DELL\Desktop\GPT train\WARNING.mp3")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Thời gian: {current_time}")
#    print(f"  - F1 Score: {f1:.2f}")
 #   print(f"  - Dự báo xu hướng: {trend} ({prediction_prob[1]:.2f})")
 #   print(f"  - Độ chính xác của mô hình meta-stacking: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    while True:
        meta_stacking_with_heikin_ashi()
        time.sleep(10)  # Chạy lại sau mỗi 10 giây
