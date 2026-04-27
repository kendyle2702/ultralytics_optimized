"""
Optimized Grad-CAM for YOLO with memory efficiency
Sử dụng EigenCAM để visualize attention maps của YOLO model.
"""

import gc

import torch
from YOLOv8_Explainer import display_images, yolov8_heatmap

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("Using CPU")

# ===========================================
# TẠI SAO DÙNG EigenCAM CHO YOLO DETECTION?
# ===========================================
# 1. EigenCAM không cần gradient → ổn định hơn, nhanh hơn
# 2. Dùng PCA/SVD để tìm principal components của activations
# 3. Không phụ thuộc vào class cụ thể → tốt cho multi-object detection
# 4. Visualize được toàn bộ regions quan trọng đồng thời
#
# CÁC THÔNG SỐ QUAN TRỌNG:
# - conf_threshold: ngưỡng confidence để filter detections (0-1)
# - layer: danh sách layer indices để extract features
#          * Layers càng sâu (gần detection head) → semantic level cao
#          * Layers càng nông → low-level features (edges, textures)
#          * Nên chọn layers ở FPN/neck để cân bằng
# - ratio: tỷ lệ top scores để tính gradient (0.02 = top 2%)
# - renormalize: có normalize CAM trong mỗi bounding box hay không
# ===========================================

print("\n" + "=" * 60)
print("INITIALIZING GRADCAM MODEL")
print("=" * 60)

# Chọn layers phù hợp với kiến trúc YOLOv8 optimized
# Model có 31 layers (0-30):
#   - Layer 15, 18: FPN upsample outputs (multi-scale features)
#   - Layer 19, 23: CBAM attention modules (focus areas)
#   - Layer 28-30: Detection neck & head
#
# Khuyến nghị: Dùng ÍT LAYERS để tránh OOM
RECOMMENDED_LAYERS = {
    "minimal": [-3],  # Chỉ layer trước detection head (nhanh nhất)
    "attention": [19, 23, -3],  # CBAM attention + detection (recommended)
    "fpn": [15, 18, -3],  # FPN multi-scale features
    "full": [12, 15, 18, 19, 23, -3],  # Toàn bộ (chậm, tốn RAM)
}

# Chọn cấu hình
CONFIG = "attention"  # Thay đổi: minimal, attention, fpn, full

model = yolov8_heatmap(
    weight="/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8_p2_cbam_scdown/best.pt",
    conf_threshold=0.25,  # Giảm xuống 0.1 nếu muốn detect nhiều objects hơn
    method="EigenCAM",  # Các options: EigenCAM, GradCAM, GradCAMPlusPlus, LayerCAM
    layer=RECOMMENDED_LAYERS[CONFIG],
    ratio=0.02,  # Chỉ lấy top 2% predictions để tính CAM
    show_box=True,  # Hiển thị bbox trên heatmap
    renormalize=False,  # False = CAM toàn ảnh, True = CAM trong từng box
)

print(f"\nUsing configuration: {CONFIG}")
print(f"Target layers: {RECOMMENDED_LAYERS[CONFIG]}")
print("\n" + "=" * 60)
print("PROCESSING IMAGE")
print("=" * 60)

# Process image
img_path = "/home/lqc/Research/Detection/ultralytics/v8_p2_cbam_scdown/9999979_00000_d_0000018.jpg"
print(f"Input image: {img_path}")

imagelist = model(img_path=img_path)

print(f"\nGenerated {len(imagelist)} visualization(s)")

# Cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("DISPLAYING RESULTS")
print("=" * 60)

# Display results
display_images(imagelist)

print("\nDone! Close the matplotlib window to exit.")
