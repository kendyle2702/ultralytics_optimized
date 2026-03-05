from ultralytics import YOLO

# Thay đường dẫn tới file model của bạn (ví dụ: best.pt)
model = YOLO("/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v12/best.pt")

# Lệnh này sẽ in ra bảng thông tin chi tiết
print(model.info())
