#!/usr/bin/env python3
"""
Training script cho YOLOv8-Visdrone trên VisDrone-people dataset
Sử dụng kiến trúc tối ưu cho small object detection

Dataset: VisDrone-people (single class)
Model: YOLOv8-Visdrone (với P2 head và CBAM attention)
"""

from ultralytics import YOLO

if __name__ == "__main__":
    # Load model từ config file (train from scratch)
    # Hoặc có thể load pretrained weights nếu muốn
    model = YOLO("ultralytics/cfg/models/v8/yolov8n-visdrone.yaml")
    
    # Nếu muốn fine-tune từ pretrained YOLOv8n:
    # model = YOLO("yolov8n.pt")
    # model.model.yaml_file = "ultralytics/cfg/models/v8/yolov8-visdrone.yaml"
    
    # Training configuration
    # Dataset: VisDrone-people
    # - Train: 6471 images
    # - Val: 548 images
    # - Single class: person (pedestrian + people merged)
    
    results = model.train(
        # Dataset configuration
        data="VisDrone-people.yaml",
        
        # Training parameters - Tối ưu cho small objects
        epochs=150,              # Nhiều epochs hơn cho model phức tạp
        patience=50,             # Early stopping
        
        # Image size và batch - QUAN TRỌNG cho small objects!
        batch=32,                # Giảm xuống nếu GPU bị OOM
        imgsz=640,              # Tăng lên 1024 để detect small objects tốt hơn
        
            
        # Checkpoints và logging
        save=True,               # Lưu checkpoint
        save_period=10,          # Lưu checkpoint mỗi 10 epochs
        project="runs/detect",   # Thư mục project
        name="yolov8_visdrone_people",  # Tên experiment
        exist_ok=False,          # Tạo folder mới nếu đã tồn tại

        
        # Device
        device=0,                # GPU 0, hoặc 'cpu' nếu không có GPU
    )
    
    # Print training results
    print("\n" + "="*60)
    print("🎉 Training completed!")
    print("="*60)
    print(f"📊 Results saved to: {results.save_dir}")
    print(f"🏆 Best model: {results.save_dir}/weights/best.pt")
    print(f"💾 Last model: {results.save_dir}/weights/last.pt")
    
    # Print metrics
    if hasattr(results, 'results_dict'):
        print("\n📈 Final metrics:")
        print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    
    print("="*60)
    print("\n💡 Để validate model:")
    print(f"   yolo detect val model={results.save_dir}/weights/best.pt data=VisDrone-people.yaml")
    print("\n💡 Để predict:")
    print(f"   yolo detect predict model={results.save_dir}/weights/best.pt source=path/to/images")
    print("="*60)

