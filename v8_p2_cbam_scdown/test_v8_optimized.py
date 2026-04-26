#!/usr/bin/env python3
"""
File đơn giản để đo metrics model v8_optimized.pt trên tập test VisDrone
Sử dụng thư viện ultralytics mặc định
"""

import json
from pathlib import Path
from ultralytics import YOLO

def main():
    # Cấu hình
    model_path = "v8_trained.pt"
    dataset_yaml = "/home/lqc/Research/Detection/ultralytics/ultralytics/cfg/datasets/VisDrone-people.yaml"
    split = "val"  # test split
    output_file = "results/v8_optimized_test_metrics.json"
    
    print("="*80)
    print("🚀 VisDrone Test Evaluation - v8_optimized.pt")
    print("="*80)
    print(f"📦 Model: {model_path}")
    print(f"📊 Dataset: VisDrone")
    print(f"🔍 Split: {split}")
    print("="*80)
    print()
    
    # Load model
    print("📦 Loading model...")
    model = YOLO(model_path)
    
    # Run validation trên test set
    print(f"🔄 Running validation on {split} set...")
    print()
    
    results = model.val(
        data=dataset_yaml,
        split=split,
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.6,
        device="0",
        verbose=True
    )
    
    # Extract metrics
    metrics = {
        "mAP50-95": float(results.box.map),      # mAP@0.5:0.95
        "mAP50": float(results.box.map50),       # mAP@0.5
        "mAP75": float(results.box.map75),       # mAP@0.75
        "precision": float(results.box.mp),      # mean precision
        "recall": float(results.box.mr),         # mean recall
        "speed_preprocess_ms": float(results.speed['preprocess']),
        "speed_inference_ms": float(results.speed['inference']),
        "speed_postprocess_ms": float(results.speed['postprocess']),
    }
    
    # Tính FPS
    total_time_ms = (metrics['speed_preprocess_ms'] + 
                     metrics['speed_inference_ms'] + 
                     metrics['speed_postprocess_ms'])
    metrics['fps'] = 1000.0 / total_time_ms if total_time_ms > 0 else 0
    
    # Print results
    print()
    print("="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    
    print("\n🚀 Speed Metrics:")
    print(f"   FPS: {metrics['fps']:.2f}")
    print(f"   Preprocess:  {metrics['speed_preprocess_ms']:.2f} ms")
    print(f"   Inference:   {metrics['speed_inference_ms']:.2f} ms")
    print(f"   Postprocess: {metrics['speed_postprocess_ms']:.2f} ms")
    print(f"   Total:       {total_time_ms:.2f} ms")
    
    print("\n📈 mAP Metrics:")
    print(f"   mAP50-95: {metrics['mAP50-95']:.4f}")
    print(f"   mAP50:    {metrics['mAP50']:.4f}")
    print(f"   mAP75:    {metrics['mAP75']:.4f}")
    
    print("\n📍 Precision & Recall:")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    
    print("\n" + "="*80)
    
    # Save results to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "config": {
            "model": model_path,
            "dataset": "VisDrone",
            "split": split,
            "imgsz": 640,
            "conf_threshold": 0.001,
            "iou_threshold": 0.6,
        },
        "metrics": metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print()

if __name__ == "__main__":
    main()

