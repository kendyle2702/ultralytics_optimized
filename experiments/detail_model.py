from ultralytics import YOLO

# Thay đường dẫn tới file model của bạn (ví dụ: best.pt)
model = YOLO("/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8/best.pt")
# model = YOLO('/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8_p2_cbam_scdown/best.pt')

# Lệnh này sẽ in ra bảng thông tin chi tiết
print(model.info())
# model.predict("/home/lqc/Research/Detection/ultralytics/v8_p2_cbam_scdown/9999979_00000_d_0000018.jpg", save=True, imgsz=640, conf=0.25)
print("\n--- DETAILED LAYERS ---")
model.info(detailed=True)
