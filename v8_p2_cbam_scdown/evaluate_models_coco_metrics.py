#!/usr/bin/env python3
"""
Comprehensive evaluation script for YOLO models on VisDrone dataset
Uses pycocotools to compute 12 standard COCO metrics

Models to evaluate:
  1. YOLOv8-base (baseline)
  2. YOLOv8-p2 (with P2 head)
  3. YOLOv8-p2-cbam (P2 + CBAM attention)
  4. YOLOv8-p2-cbam-scdown (P2 + CBAM + SCDown)
  5. YOLOv10
  6. YOLOv12

Output: 12 COCO metrics for each model saved to JSON and CSV
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
import time

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


class VisDroneEvaluator:
    """Evaluator for YOLO models on VisDrone dataset using COCO metrics."""
    
    def __init__(self, dataset_root="/home/lqc/Research/Detection/datasets"):
        """
        Initialize evaluator.
        
        Args:
            dataset_root (str): Path to datasets directory
        """
        self.dataset_root = Path(dataset_root)
        self.visdrone_path = self.dataset_root / "VisDrone"
        
        # VisDrone test set paths
        self.test_images = self.visdrone_path / "images" / "test"
        self.test_labels = self.visdrone_path / "labels" / "test"
        
        # Results directory
        self.results_dir = Path("results/coco_metrics")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # VisDrone class names (person class only for VisDrone-people)
        self.class_names = {0: "person"}  # Single class
        
        print(f"Dataset root: {self.dataset_root}")
        print(f"VisDrone path: {self.visdrone_path}")
        print(f"Test images: {self.test_images}")
        print(f"Results directory: {self.results_dir}")
    
    def create_coco_ground_truth(self):
        """
        Convert VisDrone YOLO format labels to COCO format.
        
        Returns:
            dict: COCO format ground truth
        """
        print("\n📦 Creating COCO format ground truth...")
        
        coco_gt = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        for class_id, class_name in self.class_names.items():
            coco_gt["categories"].append({
                "id": class_id + 1,  # COCO uses 1-based indexing
                "name": class_name,
                "supercategory": "object"
            })
        
        # Get all test images
        image_files = sorted(list(self.test_images.glob("*.jpg")))
        if not image_files:
            image_files = sorted(list(self.test_images.glob("*.png")))
        
        print(f"Found {len(image_files)} test images")
        
        annotation_id = 1
        
        for img_id, img_path in enumerate(tqdm(image_files, desc="Processing GT"), 1):
            # Read image to get dimensions
            from PIL import Image
            img = Image.open(img_path)
            width, height = img.size
            
            # Add image info
            coco_gt["images"].append({
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height
            })
            
            # Read corresponding label file
            label_path = self.test_labels / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
            
            # Parse YOLO format labels
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height
                    
                    # Convert to COCO format (x, y, width, height)
                    x = x_center - w / 2
                    y = y_center - h / 2
                    
                    # Add annotation
                    coco_gt["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": class_id + 1,  # COCO uses 1-based
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    annotation_id += 1
        
        # Save ground truth
        gt_path = self.results_dir / "coco_gt.json"
        with open(gt_path, 'w') as f:
            json.dump(coco_gt, f)
        
        print(f"✅ Ground truth saved: {gt_path}")
        print(f"   Images: {len(coco_gt['images'])}")
        print(f"   Annotations: {len(coco_gt['annotations'])}")
        
        return coco_gt, gt_path
    
    def count_parameters(self, model):
        """
        Count model parameters.
        
        Args:
            model: YOLO model
            
        Returns:
            dict: Parameter counts
        """
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "params_M": total_params / 1e6  # In millions
        }
    
    def measure_fps(self, model, num_warmup=10, num_iterations=100):
        """
        Measure FPS (Frames Per Second) of the model.
        
        Args:
            model: YOLO model
            num_warmup (int): Number of warmup iterations
            num_iterations (int): Number of iterations for FPS measurement
            
        Returns:
            dict: FPS metrics
        """
        print(f"   Measuring FPS (warmup={num_warmup}, iterations={num_iterations})...")
        
        # Create dummy input (640x640 as standard)
        dummy_input = torch.randn(1, 3, 640, 640)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            model.model = model.model.cuda()
        
        model.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model.model(dummy_input)
        
        # Measure FPS
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model.model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        fps = num_iterations / total_time
        latency_ms = (total_time / num_iterations) * 1000
        
        return {
            "fps": fps,
            "latency_ms": latency_ms,
            "device": "GPU" if torch.cuda.is_available() else "CPU"
        }
    
    def run_inference(self, model_path, model_name):
        """
        Run inference with YOLO model and convert to COCO format.
        Also measures parameters and FPS.
        
        Args:
            model_path (str): Path to model weights (.pt file)
            model_name (str): Name of the model
            
        Returns:
            tuple: (pred_path, model_info)
        """
        print(f"\n🔄 Running inference: {model_name}")
        print(f"   Model: {model_path}")
        
        # Load model
        model = YOLO(model_path)
        
        # Count parameters
        print(f"   Counting parameters...")
        param_info = self.count_parameters(model)
        print(f"   📊 Parameters: {param_info['params_M']:.2f}M ({param_info['total_params']:,})")
        
        # Measure FPS
        fps_info = self.measure_fps(model, num_warmup=10, num_iterations=100)
        print(f"   ⚡ FPS: {fps_info['fps']:.2f} | Latency: {fps_info['latency_ms']:.2f}ms ({fps_info['device']})")
        
        # Combine model info
        model_info = {
            **param_info,
            **fps_info
        }
        
        # Get test images
        image_files = sorted(list(self.test_images.glob("*.jpg")))
        if not image_files:
            image_files = sorted(list(self.test_images.glob("*.png")))
        
        # Run inference
        coco_predictions = []
        
        for img_id, img_path in enumerate(tqdm(image_files, desc=f"Inference {model_name}"), 1):
            # Run prediction
            results = model.predict(
                img_path,
                conf=0.001,  # Low threshold to get all detections
                iou=0.3,
                max_det=1000,
                verbose=False
            )
            
            # Convert to COCO format
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    
                    # Convert to COCO format (x, y, w, h)
                    x = float(x1)
                    y = float(y1)
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    
                    # Get confidence and class
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    coco_predictions.append({
                        "image_id": img_id,
                        "category_id": cls + 1,  # COCO uses 1-based
                        "bbox": [x, y, w, h],
                        "score": conf
                    })
        
        # Save predictions
        pred_path = self.results_dir / f"coco_pred_{model_name}.json"
        with open(pred_path, 'w') as f:
            json.dump(coco_predictions, f)
        
        print(f"✅ Predictions saved: {pred_path}")
        print(f"   Total detections: {len(coco_predictions)}")
        
        return pred_path, model_info
    
    def evaluate_coco_metrics(self, gt_path, pred_path, model_name, model_info):
        """
        Evaluate using COCO metrics.
        
        Args:
            gt_path (str): Path to ground truth JSON
            pred_path (str): Path to predictions JSON
            model_name (str): Name of the model
            model_info (dict): Model information (params, FPS)
            
        Returns:
            dict: COCO metrics + model info
        """
        print(f"\n📊 Evaluating COCO metrics: {model_name}")
        
        # Load COCO ground truth
        coco_gt = COCO(str(gt_path))
        
        # Load predictions
        coco_dt = coco_gt.loadRes(str(pred_path))
        
        # Run COCO evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract 12 COCO metrics + model info
        metrics = {
            "model": model_name,
            "params_M": model_info['params_M'],           # Parameters in millions
            "FPS": model_info['fps'],                      # Frames per second
            "latency_ms": model_info['latency_ms'],       # Latency in ms
            "AP@[.5:.95]": float(coco_eval.stats[0]),     # AP @ IoU=0.50:0.95
            "AP@.5": float(coco_eval.stats[1]),           # AP @ IoU=0.50
            "AP@.75": float(coco_eval.stats[2]),          # AP @ IoU=0.75
            "AP_small": float(coco_eval.stats[3]),        # AP for small objects
            "AP_medium": float(coco_eval.stats[4]),       # AP for medium objects
            "AP_large": float(coco_eval.stats[5]),        # AP for large objects
            "AR@1": float(coco_eval.stats[6]),            # AR given 1 detection
            "AR@10": float(coco_eval.stats[7]),           # AR given 10 detections
            "AR@100": float(coco_eval.stats[8]),          # AR given 100 detections
            "AR_small": float(coco_eval.stats[9]),        # AR for small objects
            "AR_medium": float(coco_eval.stats[10]),      # AR for medium objects
            "AR_large": float(coco_eval.stats[11]),       # AR for large objects
        }
        
        return metrics
    
    def evaluate_all_models(self, model_configs):
        """
        Evaluate all models and save results.
        
        Args:
            model_configs (dict): Dictionary of model name -> model path
            
        Returns:
            pd.DataFrame: Results table
        """
        print("="*80)
        print("🚀 Starting Comprehensive Model Evaluation")
        print("="*80)
        
        # Create ground truth
        _, gt_path = self.create_coco_ground_truth()
        
        # Store all results
        all_metrics = []
        
        # Evaluate each model
        for model_name, model_path in model_configs.items():
            print("\n" + "="*80)
            print(f"📦 Evaluating Model: {model_name}")
            print("="*80)
            
            try:
                # Check if model exists
                if not Path(model_path).exists():
                    print(f"⚠️  Model not found: {model_path}")
                    print("   Skipping...")
                    continue
                
                # Run inference and get model info
                pred_path, model_info = self.run_inference(model_path, model_name)
                
                # Evaluate COCO metrics
                metrics = self.evaluate_coco_metrics(gt_path, pred_path, model_name, model_info)
                all_metrics.append(metrics)
                
                # Save individual results
                result_file = self.results_dir / f"metrics_{model_name}.json"
                with open(result_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"✅ Results saved: {result_file}")
                
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Create results DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Save results
        self.save_results(df)
        
        return df
    
    def save_results(self, df):
        """Save results to CSV and generate comparison table."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_path = self.results_dir / f"coco_metrics_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved to: {csv_path}")
        
        # Save formatted table
        table_path = self.results_dir / f"comparison_table_{timestamp}.txt"
        with open(table_path, 'w') as f:
            f.write("="*120 + "\n")
            f.write("COCO METRICS COMPARISON - VisDrone Test Set\n")
            f.write("="*120 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n" + "="*120 + "\n")
        
        print(f"✅ Table saved to: {table_path}")
        
        # Print summary
        print("\n" + "="*120)
        print("📊 FINAL RESULTS SUMMARY")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120)
        
        # Print best model for each metric
        print("\n🏆 BEST MODELS BY METRIC:")
        print("-"*80)
        
        # For metrics where higher is better
        higher_better = ["FPS", "AP@[.5:.95]", "AP@.5", "AP@.75", "AP_small", "AP_medium", 
                        "AP_large", "AR@1", "AR@10", "AR@100", "AR_small", "AR_medium", "AR_large"]
        # For metrics where lower is better
        lower_better = ["params_M", "latency_ms"]
        
        for col in df.columns:
            if col == "model":
                continue
            
            if col in higher_better:
                best_idx = df[col].idxmax()
            elif col in lower_better:
                best_idx = df[col].idxmin()
            else:
                continue
            
            best_model = df.loc[best_idx, "model"]
            best_value = df.loc[best_idx, col]
            
            if col in ["params_M"]:
                print(f"  {col:20s}: {best_model:30s} ({best_value:.2f}M)")
            elif col in ["FPS"]:
                print(f"  {col:20s}: {best_model:30s} ({best_value:.2f} FPS)")
            elif col in ["latency_ms"]:
                print(f"  {col:20s}: {best_model:30s} ({best_value:.2f}ms)")
            else:
                print(f"  {col:20s}: {best_model:30s} ({best_value:.4f})")
        print("-"*80)
        
        # Print efficiency metrics
        print("\n⚡ EFFICIENCY SUMMARY:")
        print("-"*80)
        print(f"{'Model':<30} {'Params (M)':<12} {'FPS':<12} {'AP@.5':<12}")
        print("-"*80)
        for _, row in df.iterrows():
            print(f"{row['model']:<30} {row['params_M']:<12.2f} {row['FPS']:<12.2f} {row['AP@.5']:<12.4f}")
        print("-"*80)


def main():
    """Main function to run evaluation."""
    
    # Initialize evaluator
    evaluator = VisDroneEvaluator(dataset_root="/home/lqc/Research/Detection/datasets")
    
    # Define model configurations
    # TODO: Update these paths to your actual model paths
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
    
    # Print model paths
    print("\n" + "="*80)
    print("📝 MODEL CONFIGURATIONS")
    print("="*80)
    for name, path in model_configs.items():
        exists = "✅" if Path(path).exists() else "❌"
        print(f"{exists} {name:30s}: {path}")
    print("="*80)
    
    # Ask user to confirm or update paths
    print("\n⚠️  Please verify the model paths above are correct.")
    print("    Edit the 'model_configs' dictionary in this script if needed.")
    response = input("\nProceed with evaluation? (y/n): ")
    
    if response.lower() != 'y':
        print("Evaluation cancelled.")
        return
    
    # Run evaluation
    results_df = evaluator.evaluate_all_models(model_configs)
    
    print("\n" + "="*80)
    print("🎉 Evaluation Complete!")
    print("="*80)
    print(f"📂 Results directory: {evaluator.results_dir}")
    print("\nGenerated files:")
    print("  - coco_gt.json: Ground truth in COCO format")
    print("  - coco_pred_*.json: Predictions for each model")
    print("  - metrics_*.json: Individual model metrics")
    print("  - coco_metrics_*.csv: Combined results CSV")
    print("  - comparison_table_*.txt: Formatted comparison table")
    print("="*80)


if __name__ == "__main__":
    main()

