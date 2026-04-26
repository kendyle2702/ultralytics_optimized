
from YOLOv8_Explainer import yolov8_heatmap, display_images
import torch

# Giảm memory usage
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Chỉ dùng 1-2 layers quan trọng để tránh OOM
# Layer -3: Trước detection head (thường là feature tổng hợp cuối)
model = yolov8_heatmap(
    weight="/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8_p2_cbam_scdown/best.pt", 
        conf_threshold=0.25,  
        method = "EigenCAM",  # EigenCAM không cần gradient, ổn định với detection
        layer=[19,23],  # Chỉ dùng 1 layer để test
        ratio=0.02,  # Chỉ lấy top detections
        show_box=False,  # Hiển thị bounding boxes trên heatmap
        renormalize=False,  # Không normalize trong từng box
)

# model = yolov8_heatmap(
#     weight="/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8/best.pt", 
#         conf_threshold=0.25,  
#         method = "EigenCAM",  # EigenCAM không cần gradient, ổn định với detection
#         layer=[10, 12, 14, 16, 18, -3],  # Chỉ dùng 1 layer để test
#         ratio=0.02,  # Chỉ lấy top detections
#         show_box=False,  # Hiển thị bounding boxes trên heatmap
#         renormalize=False,  # Không normalize trong từng box
# )

imagelist = model(
    img_path="/home/lqc/Research/Detection/ultralytics/v8_p2_cbam_scdown/9999979_00000_d_0000018.jpg", 
    )

display_images(imagelist)
