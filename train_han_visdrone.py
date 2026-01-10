#!/usr/bin/env python3
"""
Training script for YOLOv8-VisDrone-HAN (Hierarchical Attention Network)
Optimized for VisDrone dataset with small object detection

Paper: "HAN-YOLO: Hierarchical Attention Network for Small Object Detection"
Contributions:
  1. Hierarchical Attention (PSA + CBAM)
  2. P2 Detection Head (stride 4)
  3. Efficient Downsampling (SCDown)
"""

from ultralytics import YOLO

if __name__ == "__main__":
    # Load HAN model architecture
    model = YOLO("ultralytics/cfg/models/v8/yolov8-visdrone-han.yaml")
    
    # Or fine-tune from pretrained YOLOv8:
    # model = YOLO("yolov8m.pt")
    # model.model.yaml_file = "ultralytics/cfg/models/v8/yolov8-visdrone-han.yaml"
    
    print("="*80)
    print("🚀 YOLOv8-VisDrone-HAN Training")
    print("="*80)
    print("Model: HAN (Hierarchical Attention Network)")
    print("Innovations:")
    print("  1. PSA + CBAM (Hierarchical Attention)")
    print("  2. P2 Detection Head (stride 4)")
    print("  3. SCDown (Efficient Downsampling)")
    print("="*80)
    
    # Training configuration optimized for VisDrone
    results = model.train(
        # Dataset
        data="VisDrone-people.yaml",
        
        # Training parameters
        epochs=150,              
        patience=50,             
        
        # Image size & batch - Important for small objects!
        batch=16,                # Adjust based on GPU memory
        imgsz=640,               # Start with 640, can increase to 1024
        
        # Optimizer - AdamW better for attention modules
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        
        # Augmentation - Conservative for small objects
        degrees=0.0,             # No rotation
        translate=0.1,
        scale=0.2,
        mosaic=1.0,
        mixup=0.1,
        
        # Training enhancements
        amp=True,                # Mixed precision
        
        # Checkpoints
        save=True,
        save_period=10,
        project="runs/han",
        name="yolov8m_han_visdrone",
        exist_ok=False,
        
        # Device
        device=0,
    )
    
    # Print results
    print("\n" + "="*80)
    print("🎉 Training completed!")
    print("="*80)
    print(f"📊 Results: {results.save_dir}")
    print(f"🏆 Best model: {results.save_dir}/weights/best.pt")
    
    if hasattr(results, 'results_dict'):
        print("\n📈 Final metrics:")
        print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    
    print("="*80)
    print("\n💡 Next steps:")
    print("  1. Validate: yolo detect val model=runs/han/.../best.pt data=VisDrone-people.yaml")
    print("  2. Ablation: Train without PSA, without CBAM, without SCDown")
    print("  3. Visualize: Generate attention maps and detection results")
    print("="*80)

