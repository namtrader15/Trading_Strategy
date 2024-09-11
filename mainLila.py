from Trend_analysis import get_trend as trend_program
from Entry import get_trend as entry_program
from binance.client import Client
from TakeProfit import meta_stacking_with_heikin_ashi
import time

# Hàm lấy giá trị tài khoản Futures
def get_account_balance(client):
    account_info = client.futures_account()
    usdt_balance = float(account_info['totalWalletBalance'])  # Số dư USDT trong tài khoản Futures
    return usdt_balance

# Hàm đóng lệnh
def close_position(client):
    symbol = 'BTCUSDT'
    
    # Lấy thông tin vị thế hiện tại
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

# Hàm cài đặt đòn bẩy cho giao dịch Futures
def set_leverage(client, symbol, leverage):
    try:
        response = client.futures_change_leverage(symbol=symbol, leverage=leverage)
        print(f"Đã cài đặt đòn bẩy {leverage}x cho {symbol}.")
    except Exception as e:
        print(f"Lỗi khi cài đặt đòn bẩy: {str(e)}")

# Hàm kiểm tra nếu có lệnh nào đang mở
def check_open_position(client, symbol):
    position_info = client.futures_position_information(symbol=symbol)
    qty = float(position_info[0]['positionAmt'])
    
    print(f"Vị thế hiện tại trong check_open_position: {qty} BTC")  # Thêm log để theo dõi trạng thái vị thế

    if qty != 0:
        print(f"Đang có lệnh mở với số lượng: {qty} BTC")
        return True  # Có lệnh mở
    return False  # Không có lệnh mở

# Hàm kiểm tra điều kiện StopLoss và TakeProfit
def check_sl_tp(client, initial_balance):
    current_balance = get_account_balance(client)
    print(f"Giá trị tài khoản hiện tại: {current_balance} USDT")

    if current_balance <= initial_balance * 0.5:
        print("Điều kiện StopLoss đạt được: Tài khoản giảm 50%. Đóng lệnh.")
        close_position(client)
        return "stop_loss"

    elif current_balance >= initial_balance * 1.8:
        print("Điều kiện TakeProfit đạt được: Tài khoản gấp 1.8. Đóng lệnh.")
        close_position(client)
        return "take_profit"
    
    return None

# Hàm thực hiện lệnh mua hoặc bán trên Binance
def place_order(client, order_type):
    symbol = 'BTCUSDT'
    usdt_balance = get_account_balance(client)
    leverage = 125  # Set đòn bẩy 1

    trading_balance = usdt_balance  * leverage * 0.8  # Giảm tỷ lệ xuống để đảm bảo đủ ký quỹ #Set volume

    ticker = client.get_symbol_ticker(symbol=symbol)
    btc_price = float(ticker['price'])
    
    # Điều chỉnh số chữ số thập phân cho quantity
    quantity = trading_balance / btc_price
    quantity = round(quantity, 3)  # Làm tròn quantity thành 3 chữ số thập phân
    
    print(f"Quantity calculated: {quantity} BTC")  # In giá trị quantity để kiểm tra

    if quantity <= 0:
        print("Lỗi: Quantity nhỏ hơn hoặc bằng 0.")
        return  # Không thực hiện giao dịch nếu quantity nhỏ hơn hoặc bằng 0

    if order_type == "buy":
        client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=quantity)
        print(f"Đã mua {quantity} BTC tại giá thị trường.")
    elif order_type == "sell":
        client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=quantity)
        print(f"Đã bán {quantity} BTC tại giá thị trường.")
    else:
        print("Lệnh không hợp lệ.")


# Main loop kiểm tra điều kiện SL/TP và thực hiện giao dịch nếu cần
def main():
    api_key = '3PxUOMsfch3HrfsuqgJJwIAD8UeSo2FnZTeoqg42gFQik4SRd5ja2IdWl16NJdQH'
    api_secret = 'urdVymoNRln92umXaKOLVVhzacnC3vjW2ANDm8z913VCqrcYfPcN8hXct76dW1Gq'
    client = Client(api_key, api_secret, tld='com', testnet=False)
    
    symbol = 'BTCUSDT'
    leverage = 125  # Set đòn bẩy 2

    # Cài đặt đòn bẩy cho cặp giao dịch
    set_leverage(client, symbol, leverage)

    initial_balance = get_account_balance(client)
    print(f"Giá trị tài khoản ban đầu: {initial_balance} USDT")

    while True:
        result = check_sl_tp(client, initial_balance)
        if result == "stop_loss" or result == "take_profit":
            continue  # Sau khi đóng lệnh do TP/SL, tiếp tục kiểm tra cơ hội entry

        trend_result = trend_program()
        entry_result = entry_program()

        # Thêm log kiểm tra giá trị trả về
        print(f"Kết quả chương trình Trend: {trend_result}")
        print(f"Kết quả chương trình Entry: {entry_result}")

        # Kiểm tra nếu đã có lệnh mở
        if check_open_position(client, symbol):
            # Chỉ ngăn mở thêm lệnh mới nếu có lệnh mở, không ngăn việc đóng vị thế
            if trend_result == entry_result:
                print("Hiện đã có lệnh mở. Không thể thực hiện thêm lệnh mới.")
                time.sleep(10)  # Đợi 10 giây trước khi kiểm tra lại
                continue

        # Kiểm tra điều kiện giao dịch
        if trend_result == 'tăng' and entry_result == 'tăng':
            print("Cả hai chương trình đều đồng ý xu hướng tăng. Thực hiện lệnh mua.")
            place_order(client, "buy")

        elif trend_result == 'giảm' and entry_result == 'giảm':
            print("Cả hai chương trình đều đồng ý xu hướng giảm. Thực hiện lệnh bán.")
            place_order(client, "sell")

        # Kiểm tra nếu trend_result khác entry_result để đóng vị thế
        elif trend_result != entry_result:
            print("Kết quả Trend và Entry không đồng nhất. Đóng tất cả các lệnh.")
            close_position(client)

        time.sleep(10)

if __name__ == "__main__":
    main()
