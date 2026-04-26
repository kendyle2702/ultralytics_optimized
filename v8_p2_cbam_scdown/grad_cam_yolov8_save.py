"""
Grad-CAM for YOLO với tùy chọn save ảnh kết quả
Phục vụ việc viết paper - tạo visualizations chất lượng cao
"""

from YOLOv8_Explainer import yolov8_heatmap
import torch
import gc
from pathlib import Path
from PIL import Image
import os

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# ===========================================
# CẤU HÌNH
# ===========================================

# Cấu hình layers
LAYER_CONFIGS = {
    "minimal": [-3],                      # Nhanh nhất
    "attention": [19, 23, -3],            # CBAM attention (recommended)
    "fpn": [15, 18, -3],                  # FPN features  
    "full": [12, 15, 18, 19, 23, -3]      # Đầy đủ nhất
}

# Cấu hình cho paper
CONFIG = {
    "model_path": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8_p2_cbam_scdown/best.pt",
    "image_path": "/home/lqc/Research/Detection/ultralytics/v8_p2_cbam_scdown/9999979_00000_d_0000018.jpg",
    "output_dir": "/home/lqc/Research/Detection/ultralytics/gradcam_output",
    "layer_config": "attention",          # Chọn: minimal, attention, fpn, full
    "conf_threshold": 0.25,
    "method": "EigenCAM",                 # EigenCAM, GradCAM, GradCAMPlusPlus, LayerCAM
    "show_box": True,
    "renormalize": False,
}

# ===========================================
# GIẢI THÍCH CAM METHODS:
# ===========================================
# 
# 1. **GradCAM**: 
#    - Dùng gradient của output class score
#    - Tốt cho classification, nhưng không ổn định với detection
#    - Cần specify target class
#
# 2. **GradCAM++**: 
#    - Cải thiện GradCAM với weighted combination
#    - Localize tốt hơn cho multiple objects
#    - Vẫn cần gradient
#
# 3. **EigenCAM** (RECOMMENDED):
#    - Dùng SVD/PCA trên activations
#    - KHÔNG cần gradient → nhanh hơn, ổn định hơn
#    - Không phụ thuộc class → tốt cho multi-object
#    - Best choice cho YOLO detection
#
# 4. **LayerCAM**:
#    - Per-pixel weighted combination
#    - Chi tiết hơn nhưng tốn compute
#
# 5. **HiResCAM**:
#    - High resolution output
#    - Rất chậm, chỉ dùng khi cần detail cao
#
# ===========================================

print("\n" + "="*70)
print(" " * 20 + "YOLO GRAD-CAM VISUALIZATION")
print("="*70)
print(f"\n📁 Model: {Path(CONFIG['model_path']).name}")
print(f"📷 Image: {Path(CONFIG['image_path']).name}")
print(f"🎯 Method: {CONFIG['method']}")
print(f"🔧 Layer config: {CONFIG['layer_config']} → {LAYER_CONFIGS[CONFIG['layer_config']]}")
print(f"📊 Confidence threshold: {CONFIG['conf_threshold']}")
print(f"💾 Output directory: {CONFIG['output_dir']}")

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# Initialize model
print("\n" + "="*70)
print("INITIALIZING MODEL...")
print("="*70)

model = yolov8_heatmap(
    weight=CONFIG['model_path'],
    conf_threshold=CONFIG['conf_threshold'],
    method=CONFIG['method'],
    layer=LAYER_CONFIGS[CONFIG['layer_config']],
    ratio=0.02,
    show_box=CONFIG['show_box'],
    renormalize=CONFIG['renormalize'],
)

# Process image
print("\n" + "="*70)
print("PROCESSING IMAGE...")
print("="*70)

imagelist = model(img_path=CONFIG['image_path'])

print(f"✅ Generated {len(imagelist)} visualization(s)")

# Save results
print("\n" + "="*70)
print("SAVING RESULTS...")
print("="*70)

img_name = Path(CONFIG['image_path']).stem
method_name = CONFIG['method'].lower()
layer_config = CONFIG['layer_config']

for idx, img_pil in enumerate(imagelist):
    # Save với tên mô tả chi tiết
    output_filename = f"{img_name}_{method_name}_{layer_config}_conf{CONFIG['conf_threshold']}.jpg"
    output_path = os.path.join(CONFIG['output_dir'], output_filename)
    
    # Save với quality cao cho paper
    img_pil.save(output_path, quality=95, dpi=(300, 300))
    
    print(f"💾 Saved: {output_filename}")
    print(f"   Full path: {output_path}")

# Cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "="*70)
print("✅ COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\n📂 Output files saved to: {CONFIG['output_dir']}")
print("\n💡 TIP: Để thử các cấu hình khác, thay đổi CONFIG['layer_config']:")
print("   - 'minimal': Nhanh nhất, 1 layer")
print("   - 'attention': CBAM attention layers (recommended cho paper)")
print("   - 'fpn': Multi-scale FPN features")
print("   - 'full': Tất cả layers (chậm nhưng detail nhất)")
print("\n💡 TIP: Thử các methods khác:")
print("   - 'EigenCAM': Không cần gradient, ổn định (recommended)")
print("   - 'GradCAM': Classic, cần gradient")
print("   - 'GradCAMPlusPlus': Improved GradCAM")
print("   - 'LayerCAM': High detail per-pixel")
print("\n" + "="*70 + "\n")

