"""
Grad-CAM Comparison Script - Tạo nhiều visualizations với các cấu hình khác nhau
Để so sánh và chọn visualization tốt nhất cho paper.
"""

import gc
import os
import time
from pathlib import Path

import torch
from YOLOv8_Explainer import yolov8_heatmap

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ===========================================
# CẤU HÌNH
# ===========================================

BASE_CONFIG = {
    "model_path": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8_p2_cbam_scdown/best.pt",
    "image_path": "/home/lqc/Research/Detection/ultralytics/v8_p2_cbam_scdown/9999979_00000_d_0000018.jpg",
    "output_dir": "/home/lqc/Research/Detection/ultralytics/gradcam_comparison",
    "conf_threshold": 0.25,
}

# Các cấu hình để so sánh
LAYER_CONFIGS = {
    "minimal": [-3],
    "attention": [19, 23, -3],
    "fpn": [15, 18, -3],
}

# Các methods để so sánh
METHODS = ["EigenCAM", "GradCAM", "LayerCAM"]

# ===========================================

print("\n" + "=" * 70)
print(" " * 15 + "GRAD-CAM COMPARISON FOR PAPER")
print("=" * 70)
print(f"\n📁 Model: {Path(BASE_CONFIG['model_path']).name}")
print(f"📷 Image: {Path(BASE_CONFIG['image_path']).name}")
print(f"📊 Confidence threshold: {BASE_CONFIG['conf_threshold']}")
print(f"💾 Output directory: {BASE_CONFIG['output_dir']}")
print(
    f"\n🔬 Will test {len(METHODS)} methods × {len(LAYER_CONFIGS)} layer configs = {len(METHODS) * len(LAYER_CONFIGS)} combinations"
)

# Create output directory
os.makedirs(BASE_CONFIG["output_dir"], exist_ok=True)

img_name = Path(BASE_CONFIG["image_path"]).stem
total_combinations = len(METHODS) * len(LAYER_CONFIGS)
current = 0

results = []

for method in METHODS:
    for layer_config_name, layer_indices in LAYER_CONFIGS.items():
        current += 1

        print("\n" + "=" * 70)
        print(f"[{current}/{total_combinations}] Processing: {method} + {layer_config_name}")
        print("=" * 70)

        try:
            start_time = time.time()

            # Initialize model
            model = yolov8_heatmap(
                weight=BASE_CONFIG["model_path"],
                conf_threshold=BASE_CONFIG["conf_threshold"],
                method=method,
                layer=layer_indices,
                ratio=0.02,
                show_box=True,
                renormalize=False,
            )

            # Process image
            imagelist = model(img_path=BASE_CONFIG["image_path"])

            # Save result
            output_filename = f"{img_name}_{method.lower()}_{layer_config_name}.jpg"
            output_path = os.path.join(BASE_CONFIG["output_dir"], output_filename)

            if imagelist and len(imagelist) > 0:
                imagelist[0].save(output_path, quality=95)

                elapsed = time.time() - start_time
                print(f"✅ Success! Saved: {output_filename}")
                print(f"⏱️  Processing time: {elapsed:.2f}s")

                results.append(
                    {
                        "method": method,
                        "config": layer_config_name,
                        "layers": str(layer_indices),
                        "time": elapsed,
                        "status": "✅ Success",
                        "file": output_filename,
                    }
                )
            else:
                print("⚠️  Warning: No output generated")
                results.append(
                    {
                        "method": method,
                        "config": layer_config_name,
                        "layers": str(layer_indices),
                        "time": 0,
                        "status": "⚠️  No output",
                        "file": "",
                    }
                )

            # Cleanup
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error: {e!s}")
            results.append(
                {
                    "method": method,
                    "config": layer_config_name,
                    "layers": str(layer_indices),
                    "time": 0,
                    "status": "❌ Error",
                    "file": "",
                }
            )

# Print summary
print("\n" + "=" * 70)
print(" " * 25 + "SUMMARY")
print("=" * 70)
print(f"\n{'Method':<20} {'Config':<15} {'Layers':<20} {'Time':<10} {'Status':<15}")
print("-" * 80)

for r in results:
    print(f"{r['method']:<20} {r['config']:<15} {r['layers']:<20} {r['time']:<10.2f}s {r['status']:<15}")

successful = sum(1 for r in results if "Success" in r["status"])
print("\n" + "=" * 70)
print(f"✅ Successfully generated: {successful}/{total_combinations} visualizations")
print(f"📂 Output directory: {BASE_CONFIG['output_dir']}")
print("\n💡 RECOMMENDATION FOR PAPER:")
print("   - EigenCAM + attention: Cân bằng giữa tốc độ và chất lượng")
print("   - EigenCAM + fpn: Tốt cho multi-scale visualization")
print("   - LayerCAM + attention: Chi tiết nhất (nếu không bị lỗi)")
print("\n" + "=" * 70 + "\n")
