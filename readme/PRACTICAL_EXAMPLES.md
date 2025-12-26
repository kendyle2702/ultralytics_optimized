# Ultralytics YOLO - Ví Dụ Thực Tế Cho Nghiên Cứu

## Ví Dụ 1: Thay Đổi Backbone từ C2f thành C3

### Bước 1: Tạo file YAML

```yaml
# cfg/models/11/yolo11_c3backbone.yaml
nc: 80
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3, [256, False]]      # ← Thay C3k2 bằng C3
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3, [512, False]]
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3, [256, False]]
  
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 2, C3, [512, False]]
  
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 2, C3, [1024, True]]
  
  - [[15, 18, 21], 1, Detect, [nc]]
```

### Bước 2: Train & Compare

```python
# train_backbone_comparison.py
from ultralytics import YOLO
import json

# Train baseline
print("Training YOLO11 (original C2f backbone)")
model_c2f = YOLO("yolo11n.yaml")
results_c2f = model_c2f.train(
    data="coco8.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    project="runs/backbone_comparison",
    name="yolo11_c2f",
)

# Train modified
print("\nTraining YOLO11 with C3 backbone")
model_c3 = YOLO("cfg/models/11/yolo11_c3backbone.yaml")
results_c3 = model_c3.train(
    data="coco8.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    project="runs/backbone_comparison",
    name="yolo11_c3",
)

# Compare
print("\n" + "="*60)
print("BACKBONE COMPARISON RESULTS")
print("="*60)

metrics = {
    "C2f (Original)": {
        "mAP50-95": results_c2f.results_dict.get('metrics/mAP50-95(B)', 0),
        "speed_inference": results_c2f.results_dict.get('speed/inference(ms)', 0),
        "params": sum(p.numel() for p in model_c2f.parameters()) / 1e6,
    },
    "C3 (Modified)": {
        "mAP50-95": results_c3.results_dict.get('metrics/mAP50-95(B)', 0),
        "speed_inference": results_c3.results_dict.get('speed/inference(ms)', 0),
        "params": sum(p.numel() for p in model_c3.parameters()) / 1e6,
    }
}

for name, data in metrics.items():
    print(f"\n{name}:")
    for key, value in data.items():
        print(f"  {key}: {value:.4f}")

# Save comparison
with open("runs/backbone_comparison/results.json", "w") as f:
    json.dump(metrics, f, indent=2)
```

---

## Ví Dụ 2: Thêm Attention vào Neck

### Bước 1: Tạo YAML với Attention

```yaml
# cfg/models/11/yolo11_attention_neck.yaml
nc: 80
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]           # Attention pool

# Enhanced NECK with attention modules
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C2fAttn, [512, 256, 8]]  # ← Attention (channels, embed_dim, num_heads)
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C2fAttn, [256, 128, 8]]  # ← Attention
  
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  
  - [[15, 18, 21], 1, Detect, [nc]]
```

### Bước 2: Training Script

```python
# train_attention.py
from ultralytics import YOLO

def train_model(config_name, project_name):
    model = YOLO(f"cfg/models/11/{config_name}.yaml")
    results = model.train(
        data="coco8.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        patience=20,
        project=f"runs/{project_name}",
        name=config_name,
    )
    return results

# Train baseline and attention version
print("Training without Attention...")
results_baseline = train_model("yolo11", "attention_study")

print("\nTraining with Attention in Neck...")
results_attention = train_model("yolo11_attention_neck", "attention_study")

# Detailed comparison
print("\n" + "="*70)
print("ATTENTION MECHANISM IMPACT")
print("="*70)

baseline_mAP = results_baseline.results_dict['metrics/mAP50-95(B)']
attention_mAP = results_attention.results_dict['metrics/mAP50-95(B)']
improvement = ((attention_mAP - baseline_mAP) / baseline_mAP) * 100

print(f"Baseline mAP@0.5-0.95: {baseline_mAP:.4f}")
print(f"With Attention mAP@0.5-0.95: {attention_mAP:.4f}")
print(f"Improvement: {improvement:+.2f}%")
```

---

## Ví Dụ 3: Custom Loss Function cho Focused Training

### Bước 1: Tạo Custom Loss

```python
# ultralytics/utils/custom_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import BboxLoss, VarifocalLoss
from .tal import TaskAlignedAssigner
from .metrics import bbox_iou

class CustomDetectionLoss:
    """
    Custom Detection Loss with class weighting
    
    Tối ưu hóa cho các lớp có dữ liệu mất cân bằng
    """
    
    def __init__(self, model, tal_topk=10, use_varifocal=True):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]
        
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.reg_max = m.reg_max
        self.device = device
        self.use_varifocal = use_varifocal
        
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, 
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        
        # Class weights - điều chỉnh cho lớp hiếm
        self.class_weights = torch.ones(self.nc, device=device)
        
    def set_class_weights(self, weights):
        """Set custom class weights"""
        self.class_weights = torch.tensor(weights, device=self.device)
    
    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = preds[1], preds[2]
        pred_scores = pred_scores.permute(0, 2, 1)  # [batch, num_anchors, num_classes]
        
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(batch["img"].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        
        # Preprocess
        gt_bboxes, gt_cls, gt_idx = [], [], []
        for i in range(batch_size):
            gi = batch["batch_idx"] == i
            bboxes = batch["bboxes"][gi]
            cls = batch["cls"][gi]
            gt_bboxes.append(bboxes)
            gt_cls.append(cls)
            gt_idx.append(gi)
        
        # Main loss computation
        anchor_points, stride_tensor = self._get_coalesced_xyxy(
            batch_size, imgsz, self.stride[0], self.device, dtype
        )
        
        # Bbox decode
        pred_bboxes = self._bbox_decode(anchor_points, pred_distri)
        
        # Task aligned assignment
        for i in range(batch_size):
            target_scores, target_bboxes, fg_mask, _ = self.assigner(
                pred_scores[i:i+1].detach().sigmoid(),
                (pred_bboxes[i:i+1].detach() * stride_tensor).type(dtype),
                anchor_points * stride_tensor,
                gt_cls[gt_idx[i]].unsqueeze(0) if len(gt_cls[gt_idx[i]]) > 0 else torch.zeros(0),
                gt_bboxes[i].unsqueeze(0) if len(gt_bboxes[i]) > 0 else torch.zeros((0, 4)),
                (torch.ones(1, *target_scores.shape[1:]) 
                 if len(gt_bboxes[i]) > 0 else torch.zeros(1, 0, 1))
            )
            
            if fg_mask.sum():
                # Classification loss with class weights
                weights = self.class_weights[gt_cls[gt_idx[i]].long()] if len(gt_cls[gt_idx[i]]) > 0 else torch.ones(1)
                loss[1] += (self.bce(pred_scores[i:i+1], target_scores) * weights.view(-1, 1)).sum()
                
                # Bbox loss
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri[i:i+1],
                    pred_bboxes[i:i+1],
                    anchor_points,
                    target_bboxes / stride_tensor,
                    target_scores,
                    max(target_scores.sum(), 1),
                    fg_mask
                )
        
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        
        return loss * batch_size, loss.detach()
    
    def _get_coalesced_xyxy(self, batch_size, imgsz, stride, device, dtype):
        """Generate anchor points"""
        # Implementation for anchor generation
        pass
    
    def _bbox_decode(self, anchor_points, pred_dist):
        """Decode bbox predictions"""
        # Implementation for bbox decoding
        pass
```

### Bước 2: Sử dụng Custom Loss trong Training

```python
# train_custom_loss.py
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.custom_loss import CustomDetectionLoss

class CustomDetectionModel(DetectionModel):
    def init_criterion(self):
        return CustomDetectionLoss(self)

class CustomDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CustomDetectionModel(
            cfg or self.args.model, 
            ch=3, 
            nc=self.data["nc"],
            verbose=verbose
        )
        if weights:
            model.load(weights)
        return model

# Training
if __name__ == "__main__":
    trainer = CustomDetectionTrainer(
        cfg=dict(
            model="yolo11n.yaml",
            data="coco8.yaml",
            epochs=100,
            imgsz=640,
            batch=16,
        )
    )
    
    # Set class weights for imbalanced dataset
    trainer.train()
```

---

## Ví Dụ 4: Architecture Search Script (NAS-inspired)

### Script để tìm kiến trúc tối ưu

```python
# architecture_search.py
"""
Simple Architecture Search for YOLO
Thử các backbone configurations khác nhau
"""

from ultralytics import YOLO
import json
from pathlib import Path

BACKBONE_TEMPLATES = {
    "light": [
        ("Conv", [32, 3, 1]),
        ("Conv", [64, 3, 2]),
        ("C2f", [64, True]),
        ("Conv", [128, 3, 2]),
        ("C2f", [128, True]),
        ("Conv", [256, 3, 2]),
        ("C2f", [256, True]),
        ("SPPF", [256, 5]),
    ],
    "medium": [
        ("Conv", [64, 3, 2]),
        ("C2f", [128, True]),
        ("Conv", [128, 3, 2]),
        ("C2f", [256, True]),
        ("Conv", [256, 3, 2]),
        ("C2f", [512, True]),
        ("Conv", [512, 3, 2]),
        ("C2f", [1024, True]),
        ("SPPF", [1024, 5]),
    ],
    "heavy": [
        ("Conv", [64, 3, 2]),
        ("C2f", [128, True]),
        ("Conv", [128, 3, 2]),
        ("C2f", [256, True]),
        ("Conv", [256, 3, 2]),
        ("C2f", [512, True]),
        ("Conv", [512, 3, 2]),
        ("C2f", [512, True]),
        ("Conv", [1024, 3, 2]),
        ("C2f", [1024, True]),
        ("C2f", [1024, True]),
        ("SPPF", [1024, 5]),
    ]
}

def generate_yaml(backbone_name, nc=80):
    """Generate YAML from template"""
    layers = BACKBONE_TEMPLATES[backbone_name]
    
    yaml_str = f"""
nc: {nc}

backbone:
"""
    
    from_idx = -1
    for i, (module, args) in enumerate(layers):
        yaml_str += f"  - [{from_idx}, 1, {module}, {args}]\n"
        from_idx = i
    
    yaml_str += """
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 5], 1, Concat, [1]]
  - [-1, 2, C2f, [512]]
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 3], 1, Concat, [1]]
  - [-1, 2, C2f, [256]]
  
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, C2f, [512]]
  
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 2, C2f, [1024]]
  
  - [[14, 17, 20], 1, Detect, [nc]]
"""
    
    return yaml_str

# Run search
results_summary = {}

for arch_name in ["light", "medium", "heavy"]:
    print(f"\n{'='*60}")
    print(f"Testing {arch_name.upper()} Architecture")
    print(f"{'='*60}")
    
    # Generate YAML
    yaml_content = generate_yaml(arch_name)
    yaml_path = Path(f"arch_search_{arch_name}.yaml")
    yaml_path.write_text(yaml_content)
    
    # Train
    model = YOLO(str(yaml_path))
    results = model.train(
        data="coco8.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project="runs/arch_search",
        name=arch_name,
    )
    
    # Store results
    params = sum(p.numel() for p in model.parameters())
    mAP = results.results_dict.get('metrics/mAP50-95(B)', 0)
    speed = results.results_dict.get('speed/inference(ms)', 0)
    
    results_summary[arch_name] = {
        "mAP50-95": mAP,
        "params_M": params / 1e6,
        "inference_ms": speed,
        "efficiency": mAP / (params / 1e6),  # mAP per million params
    }
    
    print(f"Results: mAP={mAP:.4f}, Params={params/1e6:.1f}M, Speed={speed:.2f}ms")

# Print summary
print("\n" + "="*60)
print("ARCHITECTURE SEARCH SUMMARY")
print("="*60)

for arch_name, metrics in results_summary.items():
    print(f"\n{arch_name.upper()}:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

# Save results
with open("arch_search_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

# Find best architecture
best_arch = max(results_summary.items(), key=lambda x: x[1]['efficiency'])
print(f"\n✨ Best architecture (by efficiency): {best_arch[0]}")
```

---

## Ví Dụ 5: Model Ensemble cho Nghiên Cứu

```python
# ensemble_models.py
"""
Ensemble multiple architectures for better performance
"""

from ultralytics import YOLO
import torch
import cv2
import numpy as np

class YOLOEnsemble:
    def __init__(self, model_paths):
        self.models = [YOLO(p) for p in model_paths]
    
    def predict(self, img_path, conf=0.25):
        """Ensemble prediction"""
        results_list = []
        for model in self.models:
            results = model.predict(img_path, conf=conf)
            results_list.append(results[0])
        
        # Merge predictions
        all_boxes = []
        all_confs = []
        all_classes = []
        
        for result in results_list:
            if result.boxes is not None:
                all_boxes.append(result.boxes.xyxy.cpu().numpy())
                all_confs.append(result.boxes.conf.cpu().numpy())
                all_classes.append(result.boxes.cls.cpu().numpy())
        
        if not all_boxes:
            return results_list[0]  # Return first result if no detections
        
        # Concatenate
        boxes = np.vstack(all_boxes)
        confs = np.hstack(all_confs)
        classes = np.hstack(all_classes)
        
        # NMS to remove duplicates
        from ultralytics.utils.ops import non_max_suppression
        # Implement NMS for ensemble
        
        return results_list[0]  # Return merged result

# Usage
if __name__ == "__main__":
    # Train multiple models
    models_to_train = [
        "yolo11n.yaml",
        "cfg/models/11/yolo11_attention_neck.yaml",
        "cfg/models/11/yolo11_c3backbone.yaml",
    ]
    
    trained_models = []
    for model_config in models_to_train:
        model = YOLO(model_config)
        results = model.train(
            data="coco8.yaml",
            epochs=100,
            device=0,
            project="runs/ensemble",
        )
        trained_models.append(model.model_path)
    
    # Create ensemble
    ensemble = YOLOEnsemble(trained_models)
    
    # Predict with ensemble
    ensemble_results = ensemble.predict("image.jpg")
```

---

## Ví Dụ 6: Visualization Tools cho Paper

```python
# visualization_tools.py
"""
Visualization tools for research papers
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

class ResearchVisualizer:
    @staticmethod
    def plot_architecture_comparison(results_json):
        """Plot model comparison"""
        with open(results_json) as f:
            data = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Architecture Comparison", fontsize=16, fontweight='bold')
        
        names = list(data.keys())
        
        # Plot 1: mAP
        mAP_values = [data[n]['mAP50-95'] for n in names]
        axes[0, 0].bar(names, mAP_values, color='skyblue', edgecolor='black')
        axes[0, 0].set_ylabel('mAP@0.5-0.95')
        axes[0, 0].set_title('Mean Average Precision')
        axes[0, 0].set_ylim([0, 1])
        
        # Plot 2: Parameters
        params = [data[n]['params_M'] for n in names]
        axes[0, 1].bar(names, params, color='lightcoral', edgecolor='black')
        axes[0, 1].set_ylabel('Parameters (M)')
        axes[0, 1].set_title('Model Size')
        
        # Plot 3: Speed
        speed = [data[n]['inference_ms'] for n in names]
        axes[1, 0].bar(names, speed, color='lightgreen', edgecolor='black')
        axes[1, 0].set_ylabel('Inference Time (ms)')
        axes[1, 0].set_title('Speed')
        
        # Plot 4: Pareto frontier
        axes[1, 1].scatter(params, mAP_values, s=200, alpha=0.6)
        for i, name in enumerate(names):
            axes[1, 1].annotate(name, (params[i], mAP_values[i]), 
                                ha='center', va='bottom')
        axes[1, 1].set_xlabel('Parameters (M)')
        axes[1, 1].set_ylabel('mAP@0.5-0.95')
        axes[1, 1].set_title('Efficiency Frontier')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('architecture_comparison.pdf', dpi=300, bbox_inches='tight')
        print("✓ Saved: architecture_comparison.pdf")
    
    @staticmethod
    def plot_training_curves(run_dir):
        """Plot training curves"""
        metrics_file = Path(run_dir) / "results.csv"
        
        if not metrics_file.exists():
            print(f"Metrics file not found: {metrics_file}")
            return
        
        import pandas as pd
        df = pd.read_csv(metrics_file)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
        
        # mAP curves
        axes[0, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5-0.95', marker='o')
        axes[0, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('mAP')
        axes[0, 0].set_title('Mean Average Precision')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss curves
        axes[0, 1].plot(df['epoch'], df['train/box_loss'], label='Box Loss', marker='o')
        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', marker='s')
        axes[0, 1].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', marker='^')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recall & Precision
        axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', marker='o')
        axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', marker='s')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Recall and Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='LR PG0', marker='o')
        axes[1, 1].plot(df['epoch'], df['lr/pg1'], label='LR PG1', marker='s')
        axes[1, 1].plot(df['epoch'], df['lr/pg2'], label='LR PG2', marker='^')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.pdf', dpi=300, bbox_inches='tight')
        print("✓ Saved: training_curves.pdf")

# Usage
if __name__ == "__main__":
    viz = ResearchVisualizer()
    
    # Plot comparisons
    viz.plot_architecture_comparison("results.json")
    
    # Plot training curves
    viz.plot_training_curves("runs/detect/train")
```

---

## Quick Reference Cheatsheet

### Common Training Commands

```bash
# Train with custom YAML
yolo detect train model=custom_model.yaml data=coco8.yaml epochs=100

# Resume training
yolo detect train model=last.pt resume

# Validate
yolo detect val model=best.pt data=coco8.yaml

# Predict
yolo detect predict model=best.pt source=image.jpg

# Export
yolo export model=best.pt format=onnx
```

### File Organization

```
project/
├── cfg/
│   └── models/11/
│       ├── custom_backbone.yaml
│       ├── attention_neck.yaml
│       └── optimized.yaml
├── ultralytics/
│   ├── utils/
│   │   └── custom_loss.py
│   └── nn/
│       └── modules/
│           └── custom_module.py
├── runs/
│   └── detect/
│       ├── baseline/
│       ├── custom_v1/
│       └── custom_v2/
├── train_scripts/
│   ├── train_baseline.py
│   ├── train_custom.py
│   └── train_ensemble.py
└── results/
    ├── comparison.json
    └── plots/
```

---

**Last Updated**: November 2025
**Ultralytics Version**: 8.3.228

