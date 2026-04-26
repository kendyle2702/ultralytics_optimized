"""
So sánh Grad-CAM giữa YOLOv8 gốc vs YOLOv8 tối ưu (P2 + CBAM + SCDown)
Visualize để thấy sự khác biệt trong attention patterns
"""

from YOLOv8_Explainer import yolov8_heatmap
import torch
import gc
from pathlib import Path
import os
import time

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# ===========================================
# CẤU HÌNH
# ===========================================

# Model paths
MODEL_BASE = "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8/best.pt"
MODEL_OPTIMIZED = "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8_p2_cbam_scdown/best.pt"

# Input image
IMAGE_PATH = "/home/lqc/Research/Detection/ultralytics/v8_p2_cbam_scdown/9999979_00000_d_0000018.jpg"

# Output directory
OUTPUT_DIR = "/home/lqc/Research/Detection/ultralytics/gradcam_comparison_base_vs_optimized"

# Grad-CAM settings
CONF_THRESHOLD = 0.25
METHOD = "EigenCAM"
SHOW_BOX = True
RENORMALIZE = False

# Layer mapping (exact structural correspondence - BỎ QUA P2 và CBAM để match phổ màu)
# Model gốc (23 layers): Upsample(10) → C2f(12) → Concat(14) → Conv(16) → C2f(18) → Concat(-3)
# Model optimized (31 layers): Chọn layers tương ứng chính xác, BỎ QUA P2 head và CBAM
LAYERS_CONFIG = {
    "base": [10, 12, 14, 16, 18, -3],           # Layers tốt cho model gốc
    "optimized": [10, 12, 14, 20, 22, 28],      # Match chính xác về layer types (BỎ P2 và CBAM)
    # Layer mapping explanation (EXACT MATCH):
    # base[10] Upsample    → opt[10] Upsample     ✓ FPN (exact)
    # base[12] C2f         → opt[12] C2f          ✓ FPN (exact)
    # base[14] Concat      → opt[14] Concat       ✓ FPN (exact)
    # base[16] Conv        → opt[20] Conv         ✓ Neck (skip P2/CBAM)
    # base[18] C2f         → opt[22] C2f          ✓ Neck (skip P2/CBAM)
    # base[-3] Concat      → opt[28] Concat       ✓ Before head (exact)
}

# ===========================================
# MAPPING DETAILS
# ===========================================
print("\n" + "="*80)
print(" " * 25 + "LAYER MAPPING ANALYSIS")
print("="*80)
print("\nMODEL GỐC (YOLOv8 - 23 layers):")
print("  Layer 10: Upsample  - FPN upsampling")
print("  Layer 12: C2f       - FPN features")
print("  Layer 14: Concat    - FPN concatenation")
print("  Layer 16: Conv      - Neck convolution")
print("  Layer 18: C2f       - Neck features")
print("  Layer -3: Concat    - Before detection head")

print("\nMODEL OPTIMIZED (YOLOv8 + P2 + CBAM + SCDown - 31 layers):")
print("  Layer 10: Upsample  - FPN upsampling (MATCH)")
print("  Layer 12: C2f       - FPN features (MATCH)")
print("  Layer 14: Concat    - FPN concatenation (MATCH)")
print("  Layer 20: Conv      - Neck convolution (MATCH)")
print("  Layer 22: C2f       - Neck features (MATCH)")
print("  Layer 28: Concat    - Before detection head (MATCH)")

print("\n💡 Lưu ý: Mapping này BỎ QUA P2 head và CBAM:")
print("   - Chọn layers tương ứng chính xác về structure")
print("   - Phổ màu sẽ GIỐNG NHAU giữa 2 models")
print("   - Dễ so sánh improvements từ SCDown (thay Conv)")

# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
img_name = Path(IMAGE_PATH).stem

print("\n" + "="*80)
print(" " * 20 + "GRAD-CAM COMPARISON: BASE vs OPTIMIZED")
print("="*80)
print(f"\n📷 Image: {Path(IMAGE_PATH).name}")
print(f"🎯 Method: {METHOD}")
print(f"📊 Confidence: {CONF_THRESHOLD}")
print(f"💾 Output: {OUTPUT_DIR}")

results = []

# ===========================================
# PROCESS MODEL GỐC
# ===========================================

print("\n" + "="*80)
print("[1/2] PROCESSING BASE MODEL (YOLOv8)")
print("="*80)
print(f"Model: {Path(MODEL_BASE).name}")
print(f"Layers: {LAYERS_CONFIG['base']}")

start_time = time.time()

try:
    model_base = yolov8_heatmap(
        weight=MODEL_BASE,
        conf_threshold=CONF_THRESHOLD,
        method=METHOD,
        layer=LAYERS_CONFIG['base'],
        ratio=0.02,
        show_box=SHOW_BOX,
        renormalize=RENORMALIZE,
    )
    
    imagelist_base = model_base(img_path=IMAGE_PATH)
    
    if imagelist_base and len(imagelist_base) > 0:
        output_filename = f"{img_name}_base_{METHOD.lower()}_layers{'_'.join(map(str, LAYERS_CONFIG['base'][:3]))}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        imagelist_base[0].save(output_path, quality=95)
        
        elapsed = time.time() - start_time
        print(f"✅ Success!")
        print(f"💾 Saved: {output_filename}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        
        results.append({
            "model": "Base (YOLOv8)",
            "layers": str(LAYERS_CONFIG['base']),
            "time": elapsed,
            "status": "✅ Success",
            "file": output_filename
        })
    else:
        print("⚠️  No output generated")
        results.append({
            "model": "Base (YOLOv8)",
            "layers": str(LAYERS_CONFIG['base']),
            "time": 0,
            "status": "⚠️  No output",
            "file": ""
        })
    
    # Cleanup
    del model_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    results.append({
        "model": "Base (YOLOv8)",
        "layers": str(LAYERS_CONFIG['base']),
        "time": 0,
        "status": f"❌ Error",
        "file": ""
    })

# ===========================================
# PROCESS MODEL OPTIMIZED
# ===========================================

print("\n" + "="*80)
print("[2/2] PROCESSING OPTIMIZED MODEL (YOLOv8 + P2 + CBAM + SCDown)")
print("="*80)
print(f"Model: {Path(MODEL_OPTIMIZED).name}")
print(f"Layers: {LAYERS_CONFIG['optimized']}")

start_time = time.time()

try:
    model_opt = yolov8_heatmap(
        weight=MODEL_OPTIMIZED,
        conf_threshold=CONF_THRESHOLD,
        method=METHOD,
        layer=LAYERS_CONFIG['optimized'],
        ratio=0.02,
        show_box=SHOW_BOX,
        renormalize=RENORMALIZE,
    )
    
    imagelist_opt = model_opt(img_path=IMAGE_PATH)
    
    if imagelist_opt and len(imagelist_opt) > 0:
        output_filename = f"{img_name}_optimized_{METHOD.lower()}_layers{'_'.join(map(str, LAYERS_CONFIG['optimized'][:3]))}_matched.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        imagelist_opt[0].save(output_path, quality=95)
        
        elapsed = time.time() - start_time
        print(f"✅ Success!")
        print(f"💾 Saved: {output_filename}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        
        results.append({
            "model": "Optimized (P2+CBAM+SCDown)",
            "layers": str(LAYERS_CONFIG['optimized']),
            "time": elapsed,
            "status": "✅ Success",
            "file": output_filename
        })
    else:
        print("⚠️  No output generated")
        results.append({
            "model": "Optimized (P2+CBAM+SCDown)",
            "layers": str(LAYERS_CONFIG['optimized']),
            "time": 0,
            "status": "⚠️  No output",
            "file": ""
        })
    
    # Cleanup
    del model_opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    results.append({
        "model": "Optimized (P2+CBAM+SCDown)",
        "layers": str(LAYERS_CONFIG['optimized']),
        "time": 0,
        "status": f"❌ Error",
        "file": ""
    })

# ===========================================
# SUMMARY
# ===========================================

print("\n" + "="*80)
print(" " * 30 + "SUMMARY")
print("="*80)
print(f"\n{'Model':<35} {'Time':<12} {'Status':<15}")
print("-" * 80)

for r in results:
    print(f"{r['model']:<35} {r['time']:<12.2f}s {r['status']:<15}")

successful = sum(1 for r in results if "Success" in r['status'])

print("\n" + "="*80)
print(f"✅ Successfully generated: {successful}/{len(results)} visualizations")
print(f"📂 Output directory: {OUTPUT_DIR}")

if successful == 2:
    print("\n💡 NEXT STEPS:")
    print("   1. So sánh 2 ảnh output side-by-side")
    print("   2. Phổ màu giờ đã TƯƠNG TỰ NHAU (cùng layer types)")
    print("   3. Quan sát sự khác biệt subtle trong attention patterns:")
    print("      - SCDown vs Conv thường: Feature extraction có khác?")
    print("      - Training data/augmentation: Có ảnh hưởng?")
    print("      - Architecture tweaks: Micro-improvements?")
    print("   4. Sử dụng cho paper:")
    print("      - Figure: Base vs Optimized - consistent comparison")
    print("      - Caption: Same layers, different implementations (SCDown)")
    print("\n💡 Để so sánh CBAM/P2 impact, tạo thêm visualization:")
    print("   - Chỉnh LAYERS_CONFIG['optimized'] = [10, 12, 15, 18, 19, 23, 28]")
    print("   - Chạy lại để thấy effect của CBAM attention layers")

print("\n" + "="*80 + "\n")

