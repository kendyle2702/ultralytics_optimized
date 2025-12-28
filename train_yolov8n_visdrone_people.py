#!/usr/bin/env python3
"""
Training script for YOLOv8n on VisDrone-people dataset
Gộp 2 class pedestrian và people thành 1 class person

Dataset: VisDrone-people (single class)
Model: YOLOv8n (nano - fastest, smallest)
"""

from ultralytics import YOLO

if __name__ == "__main__":
    # Load model - YOLOv8n pretrained
    # model = YOLO("yolov8n.pt")
    model = YOLO("yolov10n.pt")
    # model = YOLO("yolo12n.pt")
    
    # Training configuration
    # Với VisDrone-people dataset:
    # - Train: 6471 images
    # - Val: 548 images
    # - Single class: person (pedestrian + people merged)
    
    results = model.train(
        # Dataset configuration
        data="VisDrone-people.yaml",
        
        # Training parameters
        epochs=150,              # Số epoch: 150 cho dataset vừa phải như VisDrone
        patience=50,             # Early stopping sau 50 epochs không cải thiện
        
        # Batch và image size (giữ mặc định hoặc tùy chỉnh theo GPU)
        batch=16,                # Batch size mặc định
        imgsz=640,               # Image size mặc định của YOLO
        
        # Checkpoints và logging
        save=True,               # Lưu checkpoint
        save_period=-1,          # Không lưu checkpoint theo epoch (chỉ lưu best và last)
        project="runs/detect",   # Thư mục project
        name="yolov8n_visdrone_people",  # Tên experiment
        exist_ok=False,          # Tạo folder mới nếu đã tồn tại
    )
    
    # Print training results
    print("\n" + "="*60)
    print("🎉 Training completed!")
    print("="*60)
    print(f"📊 Results saved to: {results.save_dir}")
    print(f"🏆 Best model: {results.save_dir}/weights/best.pt")
    print(f"💾 Last model: {results.save_dir}/weights/last.pt")
    print("="*60)

