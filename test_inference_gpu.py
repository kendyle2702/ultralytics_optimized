#!/usr/bin/env python3
"""
Script đơn giản để test inference trên test set và kiểm tra GPU
"""

import os
import time
import torch
from pathlib import Path
from ultralytics import YOLO

def print_gpu_info():
    """Hiển thị thông tin GPU"""
    print("=" * 60)
    print("THÔNG TIN GPU")
    print("=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"Total GPU memory: {total_memory:.2f} GB")
        print(f"Allocated memory: {allocated_memory:.2f} GB")
        print(f"Cached memory: {cached_memory:.2f} GB")
    else:
        print("⚠️  GPU không khả dụng! Sẽ chạy trên CPU")
    print("=" * 60)
    print()

def test_inference(
    model_path="/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8/best.pt",
    test_images_dir="/home/lqc/Research/Detection/datasets/VisDrone/images/test",
    device="cuda:0",
    imgsz=640,
    conf=0.25,
    max_images=10
):
    """
    Test inference trên test set
    
    Args:
        model_path: Đường dẫn đến model weights
        test_images_dir: Thư mục chứa ảnh test
        device: Device để chạy ('cuda:0' hoặc 'cpu')
        imgsz: Kích thước ảnh
        conf: Confidence threshold
        max_images: Số ảnh tối đa để test (None = tất cả)
    """
    
    # Kiểm tra GPU
    print_gpu_info()
    
    # Load model
    print(f"📂 Đang load model từ: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model tại: {model_path}")
        return
    
    model = YOLO(model_path)
    print(f"✅ Model loaded thành công!")
    print()
    
    # Kiểm tra test images
    test_path = Path(test_images_dir)
    if not test_path.exists():
        print(f"❌ Không tìm thấy thư mục test: {test_images_dir}")
        return
    
    # Lấy danh sách ảnh test
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(test_path.glob(ext)))
    
    if len(test_images) == 0:
        print(f"❌ Không tìm thấy ảnh nào trong: {test_images_dir}")
        return
    
    print(f"📊 Tìm thấy {len(test_images)} ảnh trong test set")
    
    # Giới hạn số ảnh test nếu cần
    if max_images is not None and len(test_images) > max_images:
        test_images = test_images[:max_images]
        print(f"🎯 Sẽ test trên {max_images} ảnh đầu tiên")
    
    print()
    print("=" * 60)
    print("BẮT ĐẦU INFERENCE")
    print("=" * 60)
    
    # Đảm bảo device đúng
    if device == "cuda:0" and not torch.cuda.is_available():
        print("⚠️  CUDA không khả dụng, chuyển sang CPU")
        device = "cpu"
    
    print(f"Device: {device}")
    print(f"Image size: {imgsz}")
    print(f"Confidence threshold: {conf}")
    print()
    
    # Warm up GPU
    if "cuda" in device:
        print("🔥 Warming up GPU...")
        _ = model.predict(str(test_images[0]), device=device, verbose=False)
        torch.cuda.synchronize()
        print("✅ GPU warm-up hoàn tất")
        print()
    
    # Test inference
    total_time = 0
    total_detections = 0
    
    print("🚀 Đang chạy inference...")
    start_total = time.time()
    
    for idx, img_path in enumerate(test_images):
        # Đo thời gian inference
        start = time.time()
        
        # Run inference
        results = model.predict(
            str(img_path),
            device=device,
            imgsz=imgsz,
            conf=conf,
            verbose=False
        )
        
        # Đợi GPU hoàn tất (nếu dùng CUDA)
        if "cuda" in device:
            torch.cuda.synchronize()
        
        inference_time = time.time() - start
        total_time += inference_time
        
        # Đếm số detections
        num_detections = len(results[0].boxes)
        total_detections += num_detections
        
        # In kết quả
        print(f"  [{idx+1}/{len(test_images)}] {img_path.name}: "
              f"{num_detections} objects, {inference_time*1000:.2f}ms")
    
    end_total = time.time()
    
    # Tổng kết
    print()
    print("=" * 60)
    print("KẾT QUẢ")
    print("=" * 60)
    print(f"Tổng số ảnh: {len(test_images)}")
    print(f"Tổng thời gian: {total_time:.2f}s")
    print(f"Thời gian trung bình: {total_time/len(test_images)*1000:.2f}ms/ảnh")
    print(f"FPS: {len(test_images)/total_time:.2f}")
    print(f"Tổng số objects phát hiện: {total_detections}")
    print(f"Trung bình objects/ảnh: {total_detections/len(test_images):.2f}")
    
    # GPU memory sau khi inference
    if torch.cuda.is_available():
        print()
        print("GPU Memory sau inference:")
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Cached: {cached:.2f} GB")
    
    print("=" * 60)
    print("✅ HOÀN TẤT!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test inference và kiểm tra GPU")
    parser.add_argument("--model", type=str, default="/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8/best.pt",
                        help="Đường dẫn đến model weights")
    parser.add_argument("--test-dir", type=str, default="/home/lqc/Research/Detection/datasets/VisDrone/images/test",
                        help="Thư mục chứa ảnh test")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device để chạy (cuda:0 hoặc cpu)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Kích thước ảnh")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--max-images", type=int, default=10,
                        help="Số ảnh tối đa để test (None = tất cả)")
    
    args = parser.parse_args()
    
    test_inference(
        model_path=args.model,
        test_images_dir=args.test_dir,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        max_images=args.max_images
    )

