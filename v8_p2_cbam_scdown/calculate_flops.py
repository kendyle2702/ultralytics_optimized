#!/usr/bin/env python3
"""
Tính FLOPs cho các YOLO models
"""

import json
import torch
from pathlib import Path
from ultralytics import YOLO
from thop import profile

# Model paths
model_configs = {
    "yolov8s": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8s/best.pt",
    "yolov8-base": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8/best.pt",
    "yolov8-p2": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8_p2/best.pt",
    "yolov8-p2-cbam": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8_p2_cbam/best.pt",
    "yolov8-p2-cbam-scdown": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8_p2_cbam_scdown/best.pt",
    "yolov10": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v10/best.pt",
    "yolov12": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v12/best.pt",
    "yolov11": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v11/best.pt",
}

results = {}

print("Calculating FLOPs for all models...\n")

for model_name, model_path in model_configs.items():
    if not Path(model_path).exists():
        print(f"❌ {model_name}: Model not found")
        results[model_name] = {"error": "Model not found", "path": model_path}
        continue
    
    try:
        print(f"Processing {model_name}...")
        model = YOLO(model_path)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        
        # Calculate FLOPs using thop
        dummy_input = torch.randn(1, 3, 640, 640)
        flops, _ = profile(model.model, inputs=(dummy_input,), verbose=False)
        
        results[model_name] = {
            "path": model_path,
            "FLOPs_G": flops / 1e9,  # Convert to GFLOPs
            "params_M": total_params / 1e6,  # Parameters in millions
        }
        
        print(f"✅ {model_name}: {results[model_name]['FLOPs_G']:.2f} GFLOPs, {results[model_name]['params_M']:.2f}M params")
        
    except Exception as e:
        print(f"❌ {model_name}: Error - {e}")
        results[model_name] = {"error": str(e), "path": model_path}

# Save results
output_file = "model_flops.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")

