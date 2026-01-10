#!/usr/bin/env python3
"""
Simple script to compare YOLO baseline vs SAHI inference
Usage: python compare_sahi.py --image path/to/image.jpg
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Configuration
MODEL_PATH = "v8_trained.pt"  # Change to your model
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
SLICE_SIZE = 512
OVERLAP_RATIO = 0.1
OUTPUT_DIR = Path("comparison_outputs")


def draw_boxes(image, boxes, scores, color=(0, 255, 0)):
    """Draw bounding boxes on image"""
    img_draw = image.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_draw, f'{score:.2f}', (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img_draw


def run_baseline(image_path, model_path):
    """Run baseline YOLO inference"""
    print(f"🔄 Running baseline YOLO...")
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        str(image_path),
        imgsz=640,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )
    
    # Extract detections
    result = results[0]
    boxes = []
    scores = []
    
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
    
    print(f"   ✅ Detected {len(boxes)} objects")
    return boxes, scores


def run_sahi(image_path, model_path, slice_size=512, overlap=0.1):
    """Run SAHI inference"""
    print(f"🔄 Running SAHI (slice={slice_size}, overlap={overlap})...")
    
    # Load model with SAHI
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=CONF_THRESHOLD,
        device="cuda",
        image_size=640
    )
    
    # Run sliced prediction
    result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )
    
    # Extract detections
    boxes = []
    scores = []
    
    for obj in result.object_prediction_list:
        bbox = obj.bbox.to_xyxy()
        boxes.append(bbox)
        scores.append(obj.score.value)
    
    print(f"   ✅ Detected {len(boxes)} objects")
    return boxes, scores


def main():
    parser = argparse.ArgumentParser(description='Compare YOLO baseline vs SAHI')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--model', '-m', type=str, default=MODEL_PATH,
                       help='Model path (default: v8_trained.pt)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory (default: comparison_outputs)')
    
    args = parser.parse_args()
    
    # Setup paths
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        return
    
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🎯 Comparing YOLO Baseline vs SAHI")
    print(f"{'='*60}")
    print(f"📸 Input: {image_path}")
    print(f"🤖 Model: {args.model}")
    print(f"📁 Output: {output_dir}\n")
    
    # Load original image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Error: Could not load image")
        return
    
    h, w = image.shape[:2]
    print(f"📐 Image size: {w}×{h}\n")
    
    # Run baseline
    boxes_baseline, scores_baseline = run_baseline(image_path, args.model)
    
    # Run SAHI
    boxes_sahi, scores_sahi = run_sahi(image_path, args.model, SLICE_SIZE, OVERLAP_RATIO)
    
    # Draw results
    print(f"\n🎨 Drawing results...")
    
    # Baseline image (green boxes)
    img_baseline = draw_boxes(image, boxes_baseline, scores_baseline, color=(0, 255, 0))
    baseline_output = output_dir / f"{image_path.stem}_baseline.jpg"
    cv2.imwrite(str(baseline_output), img_baseline)
    print(f"   ✅ Saved baseline: {baseline_output}")
    
    # SAHI image (blue boxes)
    img_sahi = draw_boxes(image, boxes_sahi, scores_sahi, color=(255, 0, 0))
    sahi_output = output_dir / f"{image_path.stem}_sahi_{SLICE_SIZE}_{int(OVERLAP_RATIO*10)}.jpg"
    cv2.imwrite(str(sahi_output), img_sahi)
    print(f"   ✅ Saved SAHI: {sahi_output}")
    
    # Side-by-side comparison
    img_comparison = cv2.hconcat([img_baseline, img_sahi])
    
    # Add labels
    label_baseline = "Baseline (No SAHI)"
    label_sahi = f"SAHI ({SLICE_SIZE}/{OVERLAP_RATIO})"
    
    cv2.putText(img_comparison, label_baseline, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(img_comparison, label_sahi, (w + 20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    
    cv2.putText(img_comparison, f"Detections: {len(boxes_baseline)}", (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img_comparison, f"Detections: {len(boxes_sahi)}", (w + 20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    comparison_output = output_dir / f"{image_path.stem}_comparison.jpg"
    cv2.imwrite(str(comparison_output), img_comparison)
    print(f"   ✅ Saved comparison: {comparison_output}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Summary:")
    print(f"   Baseline detections: {len(boxes_baseline)}")
    print(f"   SAHI detections: {len(boxes_sahi)}")
    print(f"   Difference: {len(boxes_sahi) - len(boxes_baseline):+d}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

