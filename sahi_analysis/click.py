import pyautogui
import time

# --- CẤU HÌNH ---
INTERVAL = 300  # Số giây giữa mỗi lần click

print(f"Tool Auto Click đang chạy. Sẽ click mỗi {INTERVAL} giây.")
print("Để dừng lại: Nhấn tổ hợp phím 'Ctrl + C' trong cửa sổ Terminal này.")
print("LƯU Ý: Nếu mất kiểm soát chuột, hãy di nhanh chuột về 4 góc màn hình để buộc dừng.")

try:
    while True:
        
        pyautogui.click(button='right')
        # Thực hiện click chuột trái tại vị trí hiện tại của con trỏ
        pyautogui.click()
        
        # In ra màn hình để bạn biết nó đang hoạt động
        print(">> Đã click chuột trái")
        
        # Nghỉ 10 giây
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\nĐã dừng chương trình an toàn.")