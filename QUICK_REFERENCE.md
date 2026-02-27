# 🚀 Quick Reference - YOLO VisDrone Evaluation

Cheat sheet nhanh cho evaluation và training.

---

## 📝 Step-by-Step Workflow

```bash
# 1. Update model paths
nano model_paths_config.json
# OR edit directly in evaluate_models_coco_metrics.py (line 216-223)

# 2. Run evaluation (produces 12 COCO metrics)
python evaluate_models_coco_metrics.py

# 3. Generate visualizations
python visualize_coco_results.py

# 4. Check results
ls -lh results/coco_metrics/
```

---

## 🎯 Training Commands

```bash
# Train base model
python train_yolov8_visdrone_simple.py

# Or with specific config
yolo detect train \
  data=VisDrone-people.yaml \
  model=ultralytics/cfg/models/v8/yolov8-visdrone.yaml \
  epochs=150 \
  imgsz=640 \
  batch=32 \
  device=0
```

---

## 📊 Model Paths to Update

Edit in `evaluate_models_coco_metrics.py`:

```python
model_configs = {
    "yolov8-base": "runs/detect/YOUR_PATH/weights/best.pt",
    "yolov8-p2": "runs/detect/YOUR_PATH/weights/best.pt",
    "yolov8-p2-cbam": "runs/detect/YOUR_PATH/weights/best.pt",
    # ... add more models
}
```

---

## 📈 12 COCO Metrics

| Short | Full Name   | Critical for VisDrone? |
| ----- | ----------- | ---------------------- |
| AP    | AP@[.5:.95] | ✅ Primary             |
| AP50  | AP@.5       | ✅ Common              |
| AP75  | AP@.75      | ⚪ Optional            |
| APs   | AP_small    | 🔥 **CRITICAL**        |
| APm   | AP_medium   | ✅ Important           |
| APl   | AP_large    | ⚪ Less important      |
| AR1   | AR@1        | ⚪ Optional            |
| AR10  | AR@10       | ✅ Important           |
| AR100 | AR@100      | ✅ Max recall          |
| ARs   | AR_small    | 🔥 **CRITICAL**        |
| ARm   | AR_medium   | ✅ Important           |
| ARl   | AR_large    | ⚪ Less important      |

**Focus on**: AP, AP50, APs, AR100, ARs

---

## 📂 Output Files

```
results/coco_metrics/
├── coco_gt.json                      # Ground truth
├── coco_pred_MODEL.json              # Predictions per model
├── metrics_MODEL.json                # Metrics per model
├── coco_metrics_TIMESTAMP.csv        # Combined CSV
├── comparison_table_TIMESTAMP.txt    # Formatted table
└── figures/
    ├── ap_comparison.png             # AP bar chart
    ├── ap_by_size.png                # Size-specific AP
    ├── ar_comparison.png             # AR chart
    ├── metrics_heatmap.png           # All metrics
    ├── radar_chart.png               # Key metrics
    └── improvement_vs_baseline.txt   # % improvements
```

---

## 🔧 Common Issues & Fixes

### Issue 1: Model not found

```bash
# Check path exists
ls runs/detect/YOUR_PATH/weights/best.pt

# If not, update path in script
```

### Issue 2: No images found

```bash
# Verify dataset structure
ls /home/lqc/Research/Detection/datasets/VisDrone/images/test/
ls /home/lqc/Research/Detection/datasets/VisDrone/labels/test/
```

### Issue 3: CBAM KeyError

```bash
# Already fixed in tasks.py
# If still occurs, verify imports:
grep "CBAM" ultralytics/nn/tasks.py
```

---

## 📊 Good Results Threshold

```
✅ GOOD (for VisDrone small objects):
   AP@[.5:.95]: > 0.30
   AP_small:    > 0.20  ← Most important!
   AR_small:    > 0.35

⚠️ NEED IMPROVEMENT:
   AP_small:    < 0.15
   AR_small:    < 0.25
```

---

## 🎓 For Paper

### Key Metrics to Report:

1. **AP@[.5:.95]** - Primary metric
2. **AP_small** - Shows small object improvement
3. **AR_small** - Recall on small objects
4. **Improvement %** over baseline

### Template Sentence:

```
"Our method achieves X% AP@[0.5:0.95] with Y% AP_small,
representing a Z% improvement over baseline YOLOv8."
```

### Table Format:

```
Model         | AP    | AP50  | APs   | ARs
--------------|-------|-------|-------|------
YOLOv8        | 0.275 | 0.485 | 0.152 | 0.201
YOLOv8-P2     | 0.292 | 0.502 | 0.181 | 0.225
Ours (HAN)    | 0.314 | 0.538 | 0.231 | 0.278
Improvement   | +14%  | +11%  | +52%  | +38%
```

---

## 💾 Save This Command

```bash
# Complete workflow (bookmark this!)
cd /home/lqc/Research/Detection/ultralytics \
  && python evaluate_models_coco_metrics.py \
  && python visualize_coco_results.py \
  && echo "✅ Evaluation complete! Check results/coco_metrics/"
```

---

## 📞 Files to Check

- 📖 Full guide: `EVALUATION_GUIDE.md`
- 🔬 Technical details: `PAPER_CONTRIBUTIONS_ANALYSIS.md`
- 📋 Project summary: `SUMMARY_VISDRONE_PROJECT.md`
- ⚙️ Model configs: `model_paths_config.json`

---

**Created**: 2024
**Last Updated**: 2024

🎯 **Tip**: Bookmark this file for quick reference during evaluation!
