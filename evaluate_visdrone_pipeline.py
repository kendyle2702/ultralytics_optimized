#!/usr/bin/env python3
"""
Pipeline đánh giá model YOLO trên VisDrone dataset với/không SAHI
Đo metrics: FPS, mAP50, mAP50-95, mAPs, mAPm, mAPl, mAP50s, mAP50m, mAP50l

Usage:
    # Baseline (no SAHI)
    python evaluate_visdrone_pipeline.py --model yolov8n.pt --split val --no-sahi
    
    # With SAHI
    python evaluate_visdrone_pipeline.py --model yolov8n.pt --split val --sahi --slice-size 512 --overlap 0.2
"""

import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import torch

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Import COCO evaluation tools
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("⚠️  Warning: pycocotools not found. Installing...")
    import subprocess
    subprocess.run(["pip", "install", "pycocotools"], check=True)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval


class VisDroneEvaluator:
    """Pipeline đánh giá model trên VisDrone dataset với các metrics chi tiết."""
    
    def __init__(
        self,
        model_path: str,
        dataset_root: str = "/home/lqc/Research/Detection/datasets/VisDrone",
        split: str = "val",
        use_sahi: bool = False,
        slice_size: int = 640,
        overlap_ratio: float = 0.2,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "cuda",
        imgsz: int = 640,
        verbose: bool = True
    ):
        """
        Args:
            model_path: Đường dẫn đến model weights (.pt file)
            dataset_root: Đường dẫn root của VisDrone dataset
            split: 'val', 'test', hoặc 'train'
            use_sahi: Có sử dụng SAHI hay không
            slice_size: Kích thước slice cho SAHI (height = width) - ảnh gốc được cắt thành slices
            overlap_ratio: Tỷ lệ overlap cho SAHI (0.0 - 1.0)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold cho NMS
            device: Device ('', 'cpu', 'cuda', '0', '1', etc.)
            imgsz: Image size cho model input
                   - Không SAHI: Resize toàn ảnh về kích thước này
                   - Có SAHI: Mỗi slice được resize về kích thước này trước khi inference
            verbose: Print detailed logs
        """
        self.model_path = model_path
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.use_sahi = use_sahi
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.imgsz = imgsz
        self.verbose = verbose
        
        # Paths
        self.images_dir = self.dataset_root / "images" / split
        self.labels_dir = self.dataset_root / "labels" / split
        
        # Validate paths
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
        
        # Load model
        self._load_model()
        
        # Get image list
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 VisDrone Evaluation Pipeline")
            print(f"{'='*80}")
            print(f"📁 Dataset: {self.dataset_root}")
            print(f"📊 Split: {split.upper()} ({len(self.image_files)} images)")
            print(f"🤖 Model: {model_path}")
            print(f"🔧 Mode: {'SAHI' if use_sahi else 'Standard YOLO'}")
            if use_sahi:
                print(f"   - Slice size: {slice_size}×{slice_size} (cắt ảnh gốc)")
                print(f"   - Model input size: {imgsz}×{imgsz} (mỗi slice resize về kích thước này)")
                print(f"   - Overlap ratio: {overlap_ratio}")
            else:
                print(f"   - Image size: {imgsz} (resize toàn ảnh)")
            print(f"🎯 Confidence threshold: {conf_threshold}")
            print(f"🎯 IoU threshold: {iou_threshold}")
            print(f"💻 Device: {device if device else 'auto'}")
            print(f"{'='*80}\n")
    
    def _load_model(self):
        """Load YOLO model với hoặc không SAHI."""
        if self.use_sahi:
            # Load với SAHI wrapper
            if self.verbose:
                print("📦 Loading model with SAHI...")
            self.model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self.model_path,
                confidence_threshold=self.conf_threshold,
                device=self.device if self.device else "cuda" if torch.cuda.is_available() else "cpu",
                image_size=self.imgsz,  # ← THÊM: Mỗi slice sẽ được resize về kích thước này
            )
        else:
            # Load YOLO model trực tiếp
            if self.verbose:
                print("📦 Loading YOLO model...")
            self.model = YOLO(self.model_path)
    
    def _load_ground_truth(self, image_file: Path) -> List[Dict]:
        """Load ground truth labels for an image."""
        label_file = self.labels_dir / f"{image_file.stem}.txt"
        
        if not label_file.exists():
            return []
        
        # Read image size
        from PIL import Image
        img = Image.open(image_file)
        img_width, img_height = img.size
        
        ground_truths = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                cls = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # Convert to xyxy format
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                ground_truths.append({
                    'class': cls,
                    'bbox': [x1, y1, x2, y2],
                    'area': width * height
                })
        
        return ground_truths
    
    def _run_inference_sahi(self, image_path: Path) -> Tuple[List[Dict], float]:
        """Run inference với SAHI và trả về detections + inference time."""
        start_time = time.time()
        
        result = get_sliced_prediction(
            str(image_path),
            self.model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
        )
        
        inference_time = time.time() - start_time
        
        # Convert SAHI results to standard format
        detections = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox.to_xyxy()  # [x1, y1, x2, y2]
            detections.append({
                'class': obj.category.id,
                'bbox': bbox,
                'score': obj.score.value,
                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            })
        
        return detections, inference_time
    
    def _run_inference_yolo(self, image_path: Path) -> Tuple[List[Dict], float]:
        """Run inference với YOLO standard và trả về detections + inference time."""
        start_time = time.time()
        
        results = self.model.predict(
            str(image_path),
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device
        )
        
        inference_time = time.time() - start_time
        
        # Convert YOLO results to standard format
        detections = []
        result = results[0]  # First image
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                detections.append({
                    'class': int(cls),
                    'bbox': box.tolist(),
                    'score': float(score),
                    'area': float((box[2] - box[0]) * (box[3] - box[1]))  # Convert numpy float to Python float
                })
        
        return detections, inference_time
    
    def run_evaluation(self) -> Dict:
        """Chạy evaluation trên toàn bộ dataset."""
        print("🔄 Running inference on all images...\n")
        
        all_detections = []
        all_ground_truths = []
        inference_times = []
        
        # Run inference on all images
        for img_file in tqdm(self.image_files, desc="Processing"):
            # Load ground truth
            gt = self._load_ground_truth(img_file)
            
            # Run inference
            if self.use_sahi:
                detections, inf_time = self._run_inference_sahi(img_file)
            else:
                detections, inf_time = self._run_inference_yolo(img_file)
            
            inference_times.append(inf_time)
            
            # Store results with image_id
            image_id = len(all_ground_truths)
            
            for det in detections:
                all_detections.append({
                    'image_id': int(image_id),
                    'category_id': int(det['class']),
                    'bbox': [float(x) for x in det['bbox']] if not isinstance(det['bbox'], list) else det['bbox'],
                    'score': float(det['score']),
                    'area': float(det['area'])
                })
            
            for gt in gt:
                all_ground_truths.append({
                    'image_id': int(image_id),
                    'category_id': int(gt['class']),
                    'bbox': [float(x) for x in gt['bbox']],
                    'area': float(gt['area'])
                })
        
        # Calculate FPS
        avg_inference_time = np.mean(inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        print(f"\n⏱️  Average inference time: {avg_inference_time*1000:.2f} ms")
        print(f"🚀 FPS: {fps:.2f}\n")
        
        # Calculate mAP metrics
        print("📊 Calculating mAP metrics...\n")
        metrics = self._calculate_coco_metrics(all_ground_truths, all_detections)
        metrics['fps'] = fps
        metrics['avg_inference_time_ms'] = avg_inference_time * 1000
        
        return metrics
    
    def _calculate_coco_metrics(self, ground_truths: List[Dict], detections: List[Dict]) -> Dict:
        """Calculate COCO-style metrics including mAP for different object sizes."""
        
        # Create COCO format annotations
        coco_gt = {
            'images': [],
            'annotations': [],
            'categories': [{'id': 0, 'name': 'person'}]
        }
        
        # Add images
        image_ids = set([gt['image_id'] for gt in ground_truths])
        for img_id in sorted(image_ids):
            coco_gt['images'].append({
                'id': img_id,
                'file_name': f'image_{img_id}.jpg',
                'height': 1080,  # Default, will be adjusted
                'width': 1920
            })
        
        # Add ground truth annotations
        for i, gt in enumerate(ground_truths):
            bbox = gt['bbox']
            # Convert xyxy to xywh for COCO format
            x1, y1, x2, y2 = bbox
            coco_gt['annotations'].append({
                'id': i,
                'image_id': gt['image_id'],
                'category_id': gt['category_id'],
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'area': gt['area'],
                'iscrowd': 0
            })
        
        # Create COCO format detections
        coco_dt = []
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            coco_dt.append({
                'image_id': det['image_id'],
                'category_id': det['category_id'],
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': det['score'],
                'area': det['area']
            })
        
        # Save to temp files and load with COCO API
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_gt, f)
            gt_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_dt, f)
            dt_file = f.name
        
        try:
            # Load with COCO API
            coco_gt_api = COCO(gt_file)
            coco_dt_api = coco_gt_api.loadRes(dt_file) if len(coco_dt) > 0 else None
            
            if coco_dt_api is None:
                print("⚠️  No detections found!")
                return self._get_zero_metrics()
            
            # Evaluate
            coco_eval = COCOeval(coco_gt_api, coco_dt_api, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            metrics = {
                'mAP50-95': coco_eval.stats[0],  # AP @ IoU=0.50:0.95
                'mAP50': coco_eval.stats[1],      # AP @ IoU=0.50
                'mAP75': coco_eval.stats[2],      # AP @ IoU=0.75
                'mAPs': coco_eval.stats[3],       # AP for small objects
                'mAPm': coco_eval.stats[4],       # AP for medium objects
                'mAPl': coco_eval.stats[5],       # AP for large objects
                'AR@1': coco_eval.stats[6],       # AR given 1 det per image
                'AR@10': coco_eval.stats[7],      # AR given 10 det per image
                'AR@100': coco_eval.stats[8],     # AR given 100 det per image
                'ARs': coco_eval.stats[9],        # AR for small objects
                'ARm': coco_eval.stats[10],       # AR for medium objects
                'ARl': coco_eval.stats[11],       # AR for large objects
            }
            
            # Calculate mAP50 for different sizes manually if needed
            # COCO doesn't provide mAP50s, mAP50m, mAP50l directly
            # We'll compute them by filtering
            metrics['mAP50s'] = self._compute_map50_by_size(coco_gt_api, coco_dt_api, 'small')
            metrics['mAP50m'] = self._compute_map50_by_size(coco_gt_api, coco_dt_api, 'medium')
            metrics['mAP50l'] = self._compute_map50_by_size(coco_gt_api, coco_dt_api, 'large')
            
        finally:
            # Cleanup temp files
            Path(gt_file).unlink()
            Path(dt_file).unlink()
        
        return metrics
    
    def _compute_map50_by_size(self, coco_gt, coco_dt, size: str) -> float:
        """Compute mAP@50 for specific object size category."""
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        
        # Set area range based on size
        if size == 'small':
            coco_eval.params.areaRng = [[0, 32**2], [0, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
            coco_eval.params.areaRngLbl = ['all', 'small', 'medium', 'large']
            area_idx = 1
        elif size == 'medium':
            coco_eval.params.areaRng = [[0, 32**2], [0, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
            coco_eval.params.areaRngLbl = ['all', 'small', 'medium', 'large']
            area_idx = 2
        else:  # large
            coco_eval.params.areaRng = [[0, 32**2], [0, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
            coco_eval.params.areaRngLbl = ['all', 'small', 'medium', 'large']
            area_idx = 3
        
        # Set IoU threshold to 0.5
        coco_eval.params.iouThrs = [0.5]
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        # Extract AP@50 for the specific size
        precision = coco_eval.eval['precision']
        # precision has shape (T, R, K, A, M)
        # T: iou thresholds, R: recall thresholds, K: categories, A: area ranges, M: max dets
        if precision.size > 0:
            ap = precision[0, :, 0, area_idx, 2]  # IoU=0.5, cat=0, specific area, maxDet=100
            ap = ap[ap > -1]
            return np.mean(ap) if len(ap) > 0 else 0.0
        return 0.0
    
    def _get_zero_metrics(self) -> Dict:
        """Return zero metrics when no detections."""
        return {
            'mAP50-95': 0.0,
            'mAP50': 0.0,
            'mAP75': 0.0,
            'mAPs': 0.0,
            'mAPm': 0.0,
            'mAPl': 0.0,
            'mAP50s': 0.0,
            'mAP50m': 0.0,
            'mAP50l': 0.0,
            'AR@1': 0.0,
            'AR@10': 0.0,
            'AR@100': 0.0,
            'ARs': 0.0,
            'ARm': 0.0,
            'ARl': 0.0,
        }
    
    def print_results(self, metrics: Dict):
        """Print formatted results."""
        print(f"\n{'='*80}")
        print(f"📊 EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"\n🚀 Speed Metrics:")
        print(f"   FPS: {metrics['fps']:.2f}")
        print(f"   Avg inference time: {metrics['avg_inference_time_ms']:.2f} ms")
        
        print(f"\n📈 mAP Metrics (All Objects):")
        print(f"   mAP50-95: {metrics['mAP50-95']:.4f}")
        print(f"   mAP50:    {metrics['mAP50']:.4f}")
        print(f"   mAP75:    {metrics['mAP75']:.4f}")
        
        print(f"\n📊 mAP by Object Size:")
        print(f"   Small  (< 32²):      mAP50-95: {metrics['mAPs']:.4f}  |  mAP50: {metrics['mAP50s']:.4f}")
        print(f"   Medium (32²-96²):    mAP50-95: {metrics['mAPm']:.4f}  |  mAP50: {metrics['mAP50m']:.4f}")
        print(f"   Large  (> 96²):      mAP50-95: {metrics['mAPl']:.4f}  |  mAP50: {metrics['mAP50l']:.4f}")
        
        print(f"\n📍 Average Recall:")
        print(f"   AR@1:   {metrics['AR@1']:.4f}")
        print(f"   AR@10:  {metrics['AR@10']:.4f}")
        print(f"   AR@100: {metrics['AR@100']:.4f}")
        
        print(f"\n📊 AR by Object Size:")
        print(f"   ARs (small):  {metrics['ARs']:.4f}")
        print(f"   ARm (medium): {metrics['ARm']:.4f}")
        print(f"   ARl (large):  {metrics['ARl']:.4f}")
        print(f"\n{'='*80}\n")
    
    def save_results(self, metrics: Dict, output_file: str):
        """Save results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add configuration info
        results = {
            'config': {
                'model': self.model_path,
                'split': self.split,
                'use_sahi': self.use_sahi,
                'slice_size': self.slice_size if self.use_sahi else None,
                'overlap_ratio': self.overlap_ratio if self.use_sahi else None,
                'imgsz': self.imgsz if not self.use_sahi else None,
                'conf_threshold': self.conf_threshold,
                'iou_threshold': self.iou_threshold,
                'num_images': len(self.image_files)
            },
            'metrics': metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model on VisDrone dataset')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model weights (.pt file)')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, 
                        default='/home/lqc/Research/Detection/datasets/VisDrone',
                        help='Path to VisDrone dataset root')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    
    # SAHI arguments
    parser.add_argument('--sahi', action='store_true',
                        help='Use SAHI for sliced inference')
    parser.add_argument('--no-sahi', dest='sahi', action='store_false',
                        help='Do NOT use SAHI (standard YOLO inference)')
    parser.set_defaults(sahi=False)
    
    parser.add_argument('--slice-size', type=int, default=640,
                        help='Slice size for SAHI (height = width)')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='Overlap ratio for SAHI (0.0 - 1.0)')
    
    # Inference arguments
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for standard YOLO inference')
    parser.add_argument('--conf', type=float, default=0.1,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cpu, cuda, 0, 1, etc.)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='results/evaluation_results.json',
                        help='Output file for results (JSON)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed logs')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = VisDroneEvaluator(
        model_path=args.model,
        dataset_root=args.dataset,
        split=args.split,
        use_sahi=args.sahi,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        imgsz=args.imgsz,
        verbose=args.verbose
    )
    
    # Run evaluation
    metrics = evaluator.run_evaluation()
    
    # Print results
    evaluator.print_results(metrics)
    
    # Save results
    evaluator.save_results(metrics, args.output)


if __name__ == "__main__":
    main()

