from Trend_analysis import get_trend as trend_program
from Entry import get_trend as entry_program
from binance.client import Client
from TakeProfit import meta_stacking_with_heikin_ashi
import time
import sys
from playsound import playsound
# Hàm lấy giá trị tài khoản Futures
def get_account_balance(client):
    try:
        account_info = client.futures_account()
        usdt_balance = float(account_info['totalWalletBalance'])  # Số dư USDT trong tài khoản Futures
        return usdt_balance
    except Exception as e:
        print("Lỗi khi lấy thông tin tài khoản: Có thể do mất kết nối Internet.")
        print(f"Chi tiết lỗi: {str(e)}")
        return None
  
# Hàm đóng lệnh
def close_position(client):
    try:
        symbol = 'BTCUSDT'
        position_info = client.futures_position_information(symbol=symbol)
        qty = float(position_info[0]['positionAmt'])  # Số lượng vị thế hiện tại

        print(f"Vị thế hiện tại: {qty} BTC")  # Thêm thông tin log để kiểm tra vị thế

        if qty > 0:
            client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=qty)
            print(f"Đã đóng lệnh long {qty} BTC tại giá thị trường.")
        elif qty < 0:
            client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=abs(qty))
            print(f"Đã đóng lệnh short {abs(qty)} BTC tại giá thị trường.")
        else:
            print("Không có vị thế mở.")
    except Exception as e:
        print("Lỗi khi đóng lệnh: Có thể do mất kết nối Internet.")
        print(f"Chi tiết lỗi: {str(e)}")

# Hàm cài đặt đòn bẩy cho giao dịch Futures
def set_leverage(client, symbol, leverage):
    try:
        response = client.futures_change_leverage(symbol=symbol, leverage=leverage)
        print(f"Đã cài đặt đòn bẩy {leverage}x cho {symbol}.")
    except Exception as e:
        print(f"Lỗi khi cài đặt đòn bẩy: {str(e)}")

# Hàm kiểm tra nếu có lệnh nào đang mở
def check_open_position(client, symbol):
    try:
        position_info = client.futures_position_information(symbol=symbol)
        qty = float(position_info[0]['positionAmt'])
    
        print(f"Vị thế hiện tại trong check_open_position: {qty} BTC")  # Thêm log để theo dõi trạng thái vị thế

        if qty != 0:
            print(f"Đang có lệnh mở với số lượng: {qty} BTC")
            return True  # Có lệnh mở
        return False  # Không có lệnh mở
    except Exception as e:
        print("Lỗi khi kiểm tra lệnh mở: Có thể do mất kết nối Internet.")
        print(f"Chi tiết lỗi: {str(e)}")
        return False

# Hàm kiểm tra điều kiện StopLoss và TakeProfit
def check_sl_tp(client, initial_balance):
    current_balance = get_account_balance(client)
    if current_balance is None:
        return None  # Nếu không thể lấy thông tin tài khoản, dừng kiểm tra

    print(f"Giá trị tài khoản hiện tại: {current_balance} USDT")

    if current_balance <= initial_balance * 0.5:
        print("Điều kiện StopLoss đạt được: Tài khoản giảm 50%. Đóng lệnh.")
        close_position(client)
        return "stop_loss"

    elif current_balance >= initial_balance * 1.7:
        print("Điều kiện TakeProfit đạt được: Tài khoản gấp 1.7. Đóng lệnh.")
        close_position(client)
        sys.exit()
    return None

# Hàm thực hiện lệnh mua hoặc bán trên Binance
def place_order(client, order_type):
    try:
        symbol = 'BTCUSDT'
        usdt_balance = get_account_balance(client)
        if usdt_balance is None:
            return  # Nếu không thể lấy thông tin tài khoản, không thực hiện lệnh

        leverage = 125  # Set đòn bẩy 1
        trading_balance = usdt_balance * leverage * 0.8  # Giảm tỷ lệ xuống để đảm bảo đủ ký quỹ

        ticker = client.get_symbol_ticker(symbol=symbol)
        btc_price = float(ticker['price'])
    
        quantity = trading_balance / btc_price
        quantity = round(quantity, 3)

        if quantity <= 0:
            print("Lỗi: Quantity nhỏ hơn hoặc bằng 0.")
            return

        if order_type == "buy":
            client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=quantity)
            print(f"Đã mua {quantity} BTC tại giá thị trường.")
        elif order_type == "sell":
            client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=quantity)
            print(f"Đã bán {quantity} BTC tại giá thị trường.")
    except Exception as e:
        print(f"Lỗi khi thực hiện lệnh {order_type}: Có thể do mất kết nối Internet.")
        print(f"Chi tiết lỗi: {str(e)}")

# Main loop
def main():
    try:
        api_key = '4EUHKvPdCsXcOBYyRoUr5ZZ1VT1zCG5o4Ylj44hOJri9GlJl0WTriDFPDs24OVRJ'
        api_secret = 's1VvTNRO04ekqUatd0SdAaOJBRhH1379qoZ8AB0ojk4Jg9iTA7YI1f0AUOSrQU46'
        client = Client(api_key, api_secret, tld='com', testnet=False)
        
        symbol = 'BTCUSDT'
        leverage = 125

        set_leverage(client, symbol, leverage)

        initial_balance = get_account_balance(client)
        if initial_balance is None:
            return  # Dừng nếu không lấy được giá trị tài khoản ban đầu
        print(f"Giá trị tài khoản ban đầu: {initial_balance} USDT")

        while True:
            result = check_sl_tp(client, initial_balance)
            if result == "stop_loss" or result == "take_profit":
                continue

            trend_result = trend_program()
            entry_result = entry_program()

            print(f"Kết quả chương trình Trend: {trend_result}")
            print(f"Kết quả chương trình Entry: {entry_result}")

            if check_open_position(client, symbol):
                if trend_result == entry_result:
                    print("Hiện đã có lệnh mở. Không thể thực hiện thêm lệnh mới.")
                    time.sleep(10)
                    continue

            if trend_result == 'tăng' and entry_result == 'tăng':
                print("Cả hai chương trình đều đồng ý xu hướng tăng. Thực hiện lệnh mua.")
                place_order(client, "buy")
            elif trend_result == 'giảm' and entry_result == 'giảm':
                print("Cả hai chương trình đều đồng ý xu hướng giảm. Thực hiện lệnh bán.")
                place_order(client, "sell")
            elif trend_result != entry_result:
                print("Kết quả Trend và Entry không đồng nhất. Đóng tất cả các lệnh.")
                close_position(client)

            time.sleep(10)

    except Exception as e:
        print(f"Lỗi trong chương trình chính: {str(e)}")
        print("Chương trình bị lỗi. Vui lòng kiểm tra kết nối Internet.")
        playsound(r"C:\Users\DELL\Desktop\GPT train\noconnect.mp3")

if __name__ == "__main__":
    main()
