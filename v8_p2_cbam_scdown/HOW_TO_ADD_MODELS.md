# 🔧 How to Add New Models to Evaluation

Hướng dẫn nhanh để thêm models mới vào evaluation system.

---

## 📝 Method 1: Edit trong Script (Recommended)

### Step 1: Mở file evaluation script

```bash
nano evaluate_models_coco_metrics.py
# Hoặc mở bằng editor yêu thích
```

### Step 2: Tìm đến dòng ~218 (hàm `main()`)

```python
def main():
    """Main function to run evaluation."""
    
    # Initialize evaluator
    evaluator = VisDroneEvaluator(dataset_root="/home/lqc/Research/Detection/datasets")
    
    # Define model configurations
    model_configs = {
        "yolov8-base": "/path/to/yolov8_base/best.pt",
        "yolov8-p2": "/path/to/yolov8_p2/best.pt",
        # ... existing models
    }
```

### Step 3: Thêm model mới

Chỉ cần thêm 1 dòng vào dictionary `model_configs`:

```python
model_configs = {
    # Existing models
    "yolov8-base": "/home/lqc/Research/Papers/.../v8/best.pt",
    "yolov8-p2": "/home/lqc/Research/Papers/.../v8_p2/best.pt",
    
    # ✨ ADD YOUR NEW MODEL HERE ✨
    "my-new-model": "/path/to/my_model/best.pt",
    "yolov11": "/path/to/yolov11/weights/best.pt",
    "custom-yolo": "/path/to/custom/best.pt",
    # Có thể thêm bao nhiêu models tùy ý!
}
```

### Step 4: Run evaluation

```bash
python evaluate_models_coco_metrics.py
```

**Đơn giản vậy thôi!** 🎉

---

## 📝 Method 2: Config File (Alternative)

### Step 1: Edit config file

```bash
nano model_paths_config.json
```

### Step 2: Thêm model vào JSON

```json
{
  "dataset_root": "/home/lqc/Research/Detection/datasets",
  "models": {
    "yolov8-base": {
      "path": "/home/lqc/Research/.../v8/best.pt",
      "description": "YOLOv8m baseline"
    },
    
    "my-new-model": {
      "path": "/path/to/my_model/best.pt",
      "description": "My awesome model"
    }
  }
}
```

### Step 3: Load từ config (optional)

Nếu muốn dùng config file, modify script:

```python
import json

# Load config
with open('model_paths_config.json', 'r') as f:
    config = json.load(f)

# Extract model paths
model_configs = {
    name: info['path'] 
    for name, info in config['models'].items()
}
```

---

## 📊 Output cho Mỗi Model

Khi add model mới, script sẽ tự động tính toán:

### 1. Model Info:
- ✅ **Parameters** (M) - Số lượng parameters
- ✅ **FPS** - Frames per second (640x640)
- ✅ **Latency** (ms) - Thời gian inference

### 2. COCO Metrics (12 chỉ số):
- AP@[.5:.95], AP@.5, AP@.75
- AP_small, AP_medium, AP_large
- AR@1, AR@10, AR@100
- AR_small, AR_medium, AR_large

### 3. Visualizations:
- Efficiency scatter plot (FPS vs AP)
- Parameters comparison
- FPS comparison
- All standard charts

---

## 🎯 Examples

### Example 1: Add YOLOv11

```python
model_configs = {
    "yolov8-base": "/home/lqc/Research/Papers/.../v8/best.pt",
    "yolov11n": "/path/to/yolov11n/best.pt",  # ← Add this
    "yolov11s": "/path/to/yolov11s/best.pt",  # ← Add this
    "yolov11m": "/path/to/yolov11m/best.pt",  # ← Add this
}
```

### Example 2: Add Custom Models

```python
model_configs = {
    "baseline": "/path/to/baseline.pt",
    "with-attention": "/path/to/model_attention.pt",
    "with-fpn": "/path/to/model_fpn.pt",
    "full-model": "/path/to/final_model.pt",
}
```

### Example 3: Ablation Study

```python
model_configs = {
    "ablation-baseline": "/path/baseline.pt",
    "ablation-+P2": "/path/add_p2.pt",
    "ablation-+P2+CBAM": "/path/add_p2_cbam.pt",
    "ablation-+P2+CBAM+SCDown": "/path/full.pt",
}
```

---

## ⚙️ Advanced: Model Naming Conventions

### Recommended naming format:

```python
"{architecture}-{variant}-{modification}"
```

Examples:
```python
"yolov8-base"              # Clear baseline
"yolov8-p2"                # With P2 head
"yolov8-p2-cbam"           # P2 + CBAM
"yolov8-p2-cbam-scdown"    # P2 + CBAM + SCDown
"yolov10-custom"           # Custom YOLOv10
"yolov11-attention"        # YOLOv11 with attention
```

### Naming tips:
- ✅ Use lowercase with hyphens
- ✅ Be descriptive but concise
- ✅ Include key modifications
- ❌ Avoid spaces or special characters
- ❌ Don't use overly long names

---

## 🔍 Verification

### Check if model path exists:

```bash
# Method 1: Python
python -c "from pathlib import Path; print(Path('/path/to/model.pt').exists())"

# Method 2: Bash
ls -lh /path/to/model.pt
```

### Test single model quickly:

```python
from ultralytics import YOLO

# Test load
model = YOLO("/path/to/your/model.pt")
print("✅ Model loaded successfully!")

# Quick inference test
model.predict("test_image.jpg", verbose=True)
```

---

## 📊 Output File Naming

Models sẽ tạo files:

```
results/coco_metrics/
├── coco_pred_MODEL_NAME.json       # Predictions
├── metrics_MODEL_NAME.json         # Individual metrics
└── figures/                         # All models in charts
```

Ví dụ với model name `"yolov11-custom"`:
```
coco_pred_yolov11-custom.json
metrics_yolov11-custom.json
```

---

## 🚀 Quick Add Template

Copy-paste this vào script:

```python
model_configs = {
    # === EXISTING MODELS ===
    "yolov8-base": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8/best.pt",
    "yolov8-p2": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8_p2/best.pt",
    "yolov8-p2-cbam": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8_p2_cbam/best.pt",
    "yolov8-p2-cbam-scdown": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v8_p2_cbam_scdown/best.pt",
    "yolov10": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v10/best.pt",
    "yolov12": "/home/lqc/Research/Papers/Optimized_YOLO_tiny_person/v12/best.pt",
    
    # === ADD NEW MODELS BELOW ===
    # "model-name": "/path/to/model/best.pt",
}
```

---

## 💡 Tips & Best Practices

### 1. Organize by experiment:

```python
# Baseline comparisons
"v8-baseline": "/path/v8.pt",
"v10-baseline": "/path/v10.pt",
"v11-baseline": "/path/v11.pt",

# Our improvements
"v8-ours": "/path/v8_improved.pt",
"v8-ours-v2": "/path/v8_improved_v2.pt",
```

### 2. Include date/version in model names:

```python
"yolov8-v1-2024jan": "/path/...",
"yolov8-v2-2024feb": "/path/...",
```

### 3. Group related models:

```python
# Ablation study - Stage 1
"stage1-baseline": "/path/...",
"stage1-+feature1": "/path/...",
"stage1-+feature2": "/path/...",

# Ablation study - Stage 2
"stage2-baseline": "/path/...",
"stage2-optimized": "/path/...",
```

---

## 🐛 Troubleshooting

### Error: Model not found

```python
⚠️  Model not found: /path/to/model.pt
   Skipping...
```

**Solution**: Check đường dẫn file:
```bash
ls -lh /path/to/model.pt
```

### Error: Model load failed

```python
❌ Error evaluating model-name: ...
```

**Solutions**:
1. Verify model file không corrupt:
   ```bash
   file /path/to/model.pt
   ```

2. Test load manually:
   ```python
   from ultralytics import YOLO
   model = YOLO("/path/to/model.pt")
   ```

3. Check YOLO version compatibility

### Warning: Duplicate names

Nếu 2 models cùng tên, kết quả sẽ bị overwrite!

**Solution**: Dùng unique names:
```python
"yolov8-base"     # ✅ Good
"yolov8-base-v2"  # ✅ Good
"yolov8-base"     # ❌ Duplicate!
```

---

## 📈 Scaling to Many Models

Script được thiết kế để handle nhiều models:

```python
# Có thể add 10, 20, 50+ models!
model_configs = {
    f"model-{i}": f"/path/model_{i}.pt" 
    for i in range(1, 51)  # 50 models
}
```

**Note**: Mỗi model mất ~5-10 phút cho full evaluation, nên:
- 6 models: ~30-60 phút
- 10 models: ~50-100 phút
- 20 models: ~100-200 phút

---

## ✅ Checklist for Adding Models

- [ ] Model file exists và accessible
- [ ] Model name là unique
- [ ] Path dùng absolute path (recommended)
- [ ] Model tương thích với YOLO API
- [ ] Đã test load model thành công
- [ ] Updated documentation nếu cần

---

## 🎓 Summary

**To add a new model, simply:**

1. Open `evaluate_models_coco_metrics.py`
2. Find `model_configs` dictionary (~line 218)
3. Add one line: `"model-name": "/path/to/best.pt",`
4. Run: `python evaluate_models_coco_metrics.py`

**That's it! 🎉**

Script sẽ tự động:
- ✅ Load model
- ✅ Count parameters
- ✅ Measure FPS
- ✅ Run inference
- ✅ Calculate 12 COCO metrics
- ✅ Generate all visualizations
- ✅ Include in comparison tables

---

**Happy evaluating! 🚀**

