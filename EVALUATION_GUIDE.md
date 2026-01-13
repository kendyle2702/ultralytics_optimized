# 📊 COCO Metrics Evaluation Guide

Hướng dẫn chi tiết để evaluate các YOLO models trên VisDrone test set với 12 chỉ số COCO chuẩn.

---

## 📁 Files Created

```
ultralytics/
├── evaluate_models_coco_metrics.py    # Main evaluation script
├── visualize_coco_results.py          # Visualization script
├── model_paths_config.json            # Model paths configuration
└── results/
    └── coco_metrics/
        ├── coco_gt.json               # Ground truth (COCO format)
        ├── coco_pred_*.json           # Predictions for each model
        ├── metrics_*.json             # Individual model metrics
        ├── coco_metrics_*.csv         # Combined results
        ├── comparison_table_*.txt     # Formatted table
        └── figures/                   # Visualization plots
            ├── ap_comparison.png
            ├── ap_by_size.png
            ├── ar_comparison.png
            ├── metrics_heatmap.png
            ├── radar_chart.png
            └── improvement_vs_baseline.txt
```

---

## 🚀 Quick Start

### Step 1: Update Model Paths

Edit `model_paths_config.json` với đường dẫn đến các model đã train:

```json
{
  "models": {
    "yolov8-base": {
      "path": "runs/detect/yolov8m_baseline/weights/best.pt"
    },
    "yolov8-p2-cbam": {
      "path": "runs/detect/yolov8_visdrone_people/weights/best.pt"
    }
  }
}
```

Hoặc edit trực tiếp trong file `evaluate_models_coco_metrics.py` (dòng 216-223):

```python
model_configs = {
    "yolov8-base": "runs/detect/yolov8m_baseline/weights/best.pt",
    "yolov8-p2": "runs/detect/yolov8m_p2/weights/best.pt",
    # ... thêm các models khác
}
```

### Step 2: Run Evaluation

```bash
# Chạy evaluation
python evaluate_models_coco_metrics.py
```

Script sẽ:
1. ✅ Convert VisDrone labels sang COCO format
2. ✅ Run inference với mỗi model
3. ✅ Tính 12 chỉ số COCO metrics
4. ✅ Lưu kết quả vào CSV và JSON

**Thời gian ước tính:** ~5-10 phút/model (tùy số lượng test images)

### Step 3: Visualize Results

```bash
# Tạo visualizations
python visualize_coco_results.py
```

Sẽ tạo:
- 📊 Bar charts so sánh AP, AR
- 🔥 Heatmap tất cả metrics
- 🎯 Radar chart cho key metrics
- 📈 Improvement table vs baseline

---

## 📊 12 COCO Metrics Explained

| Metric | Description | Importance for VisDrone |
|--------|-------------|------------------------|
| **AP@[.5:.95]** | AP averaged over IoU thresholds [0.5:0.95] | **Primary metric** |
| **AP@.5** | AP at IoU=0.50 | Common metric, easier to achieve |
| **AP@.75** | AP at IoU=0.75 | Strict localization |
| **AP_small** | AP for small objects (area < 32²) | **Critical for VisDrone** 🔥 |
| **AP_medium** | AP for medium objects (32² < area < 96²) | Important |
| **AP_large** | AP for large objects (area > 96²) | Less common in VisDrone |
| **AR@1** | AR given 1 detection per image | Single detection quality |
| **AR@10** | AR given 10 detections per image | Multiple detections |
| **AR@100** | AR given 100 detections per image | **Max recall capability** |
| **AR_small** | AR for small objects | **Critical for VisDrone** 🔥 |
| **AR_medium** | AR for medium objects | Important |
| **AR_large** | AR for large objects | Less common |

### 🎯 Key Metrics for Paper

For VisDrone small object detection paper:
1. **AP@[.5:.95]** - Primary metric
2. **AP_small** - Shows improvement on small objects
3. **AR_small** - Recall on small objects
4. **AP@.5** - Common baseline comparison

---

## 📈 Expected Results Format

### Console Output:
```
====================================================
📊 FINAL RESULTS SUMMARY
====================================================
           model  AP@[.5:.95]   AP@.5  AP@.75  AP_small  ...
    yolov8-base        0.275   0.485   0.287     0.152
      yolov8-p2        0.292   0.502   0.305     0.181
 yolov8-p2-cbam        0.314   0.538   0.328     0.231
====================================================
```

### CSV File:
```csv
model,AP@[.5:.95],AP@.5,AP@.75,AP_small,AP_medium,AP_large,AR@1,AR@10,AR@100,AR_small,AR_medium,AR_large
yolov8-base,0.275,0.485,0.287,0.152,0.285,0.412,0.245,0.358,0.402,0.201,0.375,0.485
```

---

## 🔧 Configuration Options

### Dataset Path

Edit trong `evaluate_models_coco_metrics.py`:

```python
evaluator = VisDroneEvaluator(dataset_root="/home/lqc/Research/Detection/datasets")
```

### Inference Parameters

Trong method `run_inference()` (line 147):

```python
results = model.predict(
    img_path,
    conf=0.001,    # Confidence threshold (low to get all detections)
    iou=0.3,       # NMS IoU threshold
    max_det=1000,  # Maximum detections per image
    verbose=False
)
```

Adjust these based on your needs:
- `conf=0.001`: Very low để evaluate recall capacity
- `conf=0.25`: Standard threshold cho production
- `iou=0.3`: Lower for VisDrone (nhiều objects overlapping)

---

## 📊 Interpreting Results

### For Paper Writing:

#### 1. Baseline Comparison
```python
"Our YOLOv8-P2-CBAM achieves 0.314 AP@[.5:.95], 
outperforming baseline YOLOv8 (0.275) by 14.2%"
```

#### 2. Small Object Performance
```python
"On small objects (< 32² pixels), our method achieves 
0.231 AP_small, a significant 51.3% improvement over 
baseline (0.152 AP_small)"
```

#### 3. Recall Improvement
```python
"Our model demonstrates superior recall capability with 
AR@100 of 0.445, indicating better detection of all 
ground truth objects"
```

### 🎯 Good vs Bad Results

**Good Results** (for small object detection):
- ✅ AP@[.5:.95] > 0.30
- ✅ AP_small > 0.20 (critical!)
- ✅ AR_small > 0.35
- ✅ Improvement over baseline > 10%

**Need Improvement**:
- ❌ AP_small < 0.15
- ❌ Large gap between AP@.5 and AP@.75 (poor localization)
- ❌ AR@100 too low (missing detections)

---

## 🐛 Troubleshooting

### Error: "Model not found"
```bash
⚠️  Model not found: runs/detect/xxx/weights/best.pt
```

**Solution**: Verify đường dẫn model trong config file hoặc script.

### Error: "No images found"
```bash
Found 0 test images
```

**Solution**: 
- Check dataset path: `/home/lqc/Research/Detection/datasets/VisDrone/images/test/`
- Make sure images có extension `.jpg` hoặc `.png`

### Error: "Division by zero" trong COCO eval
```bash
Warning: No detections/ground truths found
```

**Solution**: 
- Check labels có đúng format YOLO không
- Verify class IDs match (0-based for YOLO, 1-based for COCO)

### Memory Error
```bash
CUDA out of memory
```

**Solution**: 
- Reduce batch size trong inference
- Process images sequentially (already done in script)
- Use CPU inference: `device='cpu'`

---

## 📝 Advanced Usage

### Evaluate Single Model

```python
from evaluate_models_coco_metrics import VisDroneEvaluator

evaluator = VisDroneEvaluator()
_, gt_path = evaluator.create_coco_ground_truth()

# Run single model
pred_path = evaluator.run_inference(
    "runs/detect/my_model/weights/best.pt",
    "my_model"
)

# Get metrics
metrics = evaluator.evaluate_coco_metrics(gt_path, pred_path, "my_model")
print(metrics)
```

### Custom Metrics Selection

Edit `visualize_coco_results.py` để chỉ plot metrics cần thiết:

```python
# For paper, focus on these:
key_metrics = ["AP@[.5:.95]", "AP@.5", "AP_small", "AR_small"]
```

### Export for LaTeX

```python
import pandas as pd

df = pd.read_csv("results/coco_metrics/coco_metrics_20240101_120000.csv")

# Export to LaTeX table
latex_table = df.to_latex(index=False, float_format="%.3f")
with open("results/table.tex", "w") as f:
    f.write(latex_table)
```

---

## 📚 Integration với Paper

### Table for Paper

```latex
\begin{table}[t]
\centering
\caption{COCO Metrics Comparison on VisDrone Test Set}
\label{tab:coco_metrics}
\begin{tabular}{lcccccc}
\toprule
Model & AP & AP$_{50}$ & AP$_{75}$ & AP$_S$ & AR$_{100}$ & AR$_S$ \\
\midrule
YOLOv8 & 0.275 & 0.485 & 0.287 & 0.152 & 0.402 & 0.201 \\
YOLOv8-P2 & 0.292 & 0.502 & 0.305 & 0.181 & 0.421 & 0.225 \\
\textbf{Ours (HAN)} & \textbf{0.314} & \textbf{0.538} & \textbf{0.328} & \textbf{0.231} & \textbf{0.445} & \textbf{0.278} \\
\bottomrule
\end{tabular}
\end{table}
```

### Citation

```bibtex
@misc{cocoapi,
  author = {COCO Consortium},
  title = {COCO: Common Objects in Context},
  year = {2014},
  url = {https://cocodataset.org/}
}
```

---

## ✅ Checklist for Paper

- [ ] Run evaluation on all models
- [ ] Generate all visualizations
- [ ] Create comparison table
- [ ] Calculate improvement percentages
- [ ] Verify AP_small shows improvement (most important!)
- [ ] Check AR_small (recall on small objects)
- [ ] Save figures for paper (300 DPI)
- [ ] Export table to LaTeX format
- [ ] Document any special configurations used

---

## 🎯 Paper Writing Tips

### Abstract:
```
"Evaluated on VisDrone test set, our method achieves X% AP@[0.5:0.95]
with particularly strong performance on small objects (Y% AP_small),
representing a Z% improvement over baseline YOLOv8."
```

### Results Section:
```
"Table X presents comprehensive COCO metrics. Our method excels at
detecting small objects, achieving 0.231 AP_small compared to 0.152
for baseline YOLOv8, a 51.3% improvement."
```

### Discussion:
```
"The substantial improvement in AR_small (0.278 vs 0.201) demonstrates
our hierarchical attention mechanism effectively captures fine-grained
features critical for small object detection in aerial imagery."
```

---

**Good luck with your evaluation and paper! 🚀📊**

