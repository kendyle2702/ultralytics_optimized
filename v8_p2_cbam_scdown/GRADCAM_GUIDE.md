# Hướng Dẫn Sử Dụng Grad-CAM cho YOLO Detection

## 📚 Giới Thiệu về Grad-CAM

### Grad-CAM là gì?
**Gradient-weighted Class Activation Mapping (Grad-CAM)** là kỹ thuật visualization giúp hiểu được model CNN "nhìn vào đâu" khi đưa ra quyết định. Nó tạo ra heatmap highlighting các vùng quan trọng trong ảnh.

### Cách Hoạt Động
1. **Forward pass**: Đưa ảnh qua model, lưu activations của target layers
2. **Backward pass**: Tính gradient của output score đối với activations
3. **Weighted combination**: Kết hợp channels với trọng số từ gradients
4. **ReLU + Upsampling**: Loại bỏ negative values và resize về kích thước ảnh gốc

## 🎯 Tại Sao Dùng EigenCAM cho YOLO?

### So Sánh Các Methods

| Method | Cần Gradient? | Tốc độ | Ổn định | Phù hợp Detection? | Khi nào dùng? |
|--------|--------------|--------|---------|-------------------|---------------|
| **EigenCAM** ✅ | ❌ Không | ⚡ Nhanh | ✅ Cao | ✅✅✅ Tốt nhất | **RECOMMENDED** cho YOLO |
| GradCAM | ✅ Có | 🐢 Trung bình | ⚠️ Trung bình | ⚠️ OK | Classification tasks |
| GradCAM++ | ✅ Có | 🐢 Trung bình | ⚠️ Trung bình | ✅ Tốt | Multiple objects |
| LayerCAM | ✅ Có | 🐌 Chậm | ⚠️ Thấp | ✅ Tốt | Cần detail cao |
| HiResCAM | ✅ Có | 🐌🐌 Rất chậm | ⚠️ Thấp | ✅ Tốt | High-res visualization |

### Tại Sao EigenCAM?

1. **Không cần gradient** 
   - Dùng SVD/PCA trên activations
   - Nhanh hơn, ít memory hơn
   - Không bị vanishing/exploding gradients

2. **Không phụ thuộc class**
   - Visualize toàn bộ regions quan trọng
   - Tốt cho multi-object detection
   - Không cần chọn target class

3. **Ổn định với multi-task learning**
   - YOLO phải predict: objectness + classes + bounding boxes
   - Gradient-based methods có thể không ổn định
   - EigenCAM luôn cho kết quả consistent

## 🔧 Các Thông Số Quan Trọng

### 1. `weight` (string)
- Đường dẫn đến model checkpoint (.pt file)
- **Ví dụ**: `"/path/to/best.pt"`

### 2. `conf_threshold` (float, 0-1)
- Ngưỡng confidence để filter detections
- **Giá trị thấp** (0.1): Detect nhiều objects hơn, có thể nhiều false positives
- **Giá trị cao** (0.5): Chỉ giữ detections chắc chắn
- **Recommended**: 0.25 - 0.35 cho general use

### 3. `method` (string)
- Phương pháp tính CAM
- **Options**: "EigenCAM", "GradCAM", "GradCAMPlusPlus", "LayerCAM", "HiResCAM"
- **Recommended**: "EigenCAM" cho YOLO

### 4. `layer` (list of integers)
- Danh sách indices của layers để extract features
- **Rất quan trọng!** Chọn sai layers → kết quả kém hoặc OOM error

#### Cách Chọn Layers

```python
# Ví dụ: YOLOv8 optimized có 31 layers (0-30)

# Option 1: Minimal (nhanh nhất)
layer = [-3]  # Chỉ layer trước detection head

# Option 2: Attention (recommended)
layer = [19, 23, -3]  # CBAM attention layers + detection

# Option 3: FPN multi-scale
layer = [15, 18, -3]  # FPN outputs ở các scales khác nhau

# Option 4: Full (chậm, detail)
layer = [12, 15, 18, 19, 23, -3]  # Tất cả các layers quan trọng
```

**Nguyên tắc**:
- Layers **càng sâu** (gần head): Semantic-level cao, concepts phức tạp
- Layers **càng nông** (gần input): Low-level features (edges, textures)
- **FPN/Neck layers**: Cân bằng giữa semantic và spatial information
- **Attention layers** (CBAM, etc.): Highlight focus areas của model

#### Kiểm Tra Layer Indices

```python
from ultralytics import YOLO

model = YOLO("path/to/model.pt")
for i, layer in enumerate(model.model.model):
    print(f"Layer {i}: {layer.__class__.__name__}")
```

### 5. `ratio` (float, 0-1)
- Tỷ lệ top scores để tính gradient
- **0.02** = top 2% predictions
- Giảm xuống nếu có ít detections
- Tăng lên nếu có quá nhiều detections

### 6. `show_box` (boolean)
- `True`: Vẽ bounding boxes lên heatmap
- `False`: Chỉ hiển thị heatmap
- **Recommended**: `True` cho paper (dễ interpret)

### 7. `renormalize` (boolean)
- `False`: Normalize CAM toàn ảnh (recommended)
- `True`: Normalize CAM trong từng bounding box riêng biệt
- **Trade-off**: 
  - `False`: Dễ so sánh importance giữa các objects
  - `True`: Mỗi object có detail riêng rõ hơn

## 🚀 Scripts Có Sẵn

### 1. `grad_cam_yolov8.py`
- Script cơ bản, display kết quả
- Tốt để test nhanh

### 2. `grad_cam_yolov8_optimized.py` ⭐
- Có memory management
- Verbose output để debug
- Nhiều layer configs preset

### 3. `grad_cam_yolov8_save.py` ⭐⭐
- **Recommended cho paper**
- Save ảnh với quality cao (95%, 300 DPI)
- Tên file descriptive
- Dễ config

### 4. `grad_cam_comparison.py` ⭐⭐⭐
- **Best for research**
- Tạo nhiều visualizations một lúc
- So sánh methods và configs
- Summary table với timing

## 📝 Ví Dụ Sử Dụng

### Cơ Bản

```python
from YOLOv8_Explainer import yolov8_heatmap, display_images

model = yolov8_heatmap(
    weight="best.pt",
    conf_threshold=0.25,
    method="EigenCAM",
    layer=[-3],
    show_box=True,
)

images = model(img_path="test.jpg")
display_images(images)
```

### Lưu Kết Quả Cho Paper

```bash
# Chỉnh sửa CONFIG trong grad_cam_yolov8_save.py
python grad_cam_yolov8_save.py
```

### So Sánh Nhiều Cấu Hình

```bash
# Tạo grid so sánh các methods và layer configs
python grad_cam_comparison.py
```

## ⚠️ Xử Lý Lỗi Thường Gặp

### 1. ImportError: cannot import name 'attempt_load_weights'

**Nguyên nhân**: YOLOv8_Explainer không tương thích với ultralytics mới

**Giải pháp**: Đã fix trong `/home/lqc/miniconda3/envs/tracking-service/lib/python3.12/site-packages/YOLOv8_Explainer/core.py`

```python
# Thay đổi
from ultralytics.nn.tasks import attempt_load_weights
# Thành
from ultralytics.nn.tasks import load_checkpoint

# Và
model = attempt_load_weights(weight, device)
# Thành  
model, ckpt = load_checkpoint(weight, device)
```

### 2. ImportError: cannot import name 'non_max_suppression'

**Nguyên nhân**: Function đã bị move sang module khác

**Giải pháp**: 
```python
# Thay đổi
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy
# Thành
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy
```

### 3. Script Treo Máy / OOM Error

**Nguyên nhân**: 
- Quá nhiều layers
- Model quá lớn
- Insufficient GPU memory

**Giải pháp**:
```python
# 1. Giảm số layers
layer = [-3]  # Thay vì [12, 15, 18, 19, 23, -3]

# 2. Clear cache trước khi chạy
import torch
torch.cuda.empty_cache()

# 3. Dùng CPU nếu cần
device = torch.device("cpu")
```

### 4. IndexError: list index out of range

**Nguyên nhân**: Layer index không tồn tại

**Giải pháp**: Kiểm tra số layers của model
```python
from ultralytics import YOLO
model = YOLO("best.pt")
print(f"Total layers: {len(model.model.model)}")
```

## 📊 Để Viết Paper

### Visualization Recommendations

1. **Main Figure**: Dùng EigenCAM + attention layers
   - Cân bằng giữa clarity và detail
   - Stable và reproducible

2. **Comparison Figure**: So sánh EigenCAM vs GradCAM
   - Chứng minh EigenCAM tốt hơn cho detection
   - Highlight stability

3. **Ablation Study**: So sánh các layer configs
   - Minimal vs FPN vs Full
   - Show impact of layer selection

### Metrics to Report

- **Processing time**: So sánh tốc độ các methods
- **Memory usage**: GPU memory consumption
- **Visualization quality**: Subjective evaluation
- **Localization accuracy**: IoU với ground truth attention

### Caption Template

```
Figure X: Grad-CAM visualization of YOLOv8 optimized model. 
(Left) Original image with detections. 
(Right) EigenCAM heatmap overlaid with bounding boxes. 
The heatmap highlights regions where the model focuses attention for object detection.
Red indicates high importance, blue indicates low importance.
Visualization generated from layers [19, 23, -3] (CBAM attention modules and 
detection head) using EigenCAM method with confidence threshold 0.25.
```

## 🔬 Hiểu Kết Quả

### Màu Sắc Heatmap
- **Đỏ/Vàng**: Model chú ý nhiều (important regions)
- **Xanh/Tím**: Model chú ý ít
- **Green**: Trung bình

### Good Visualization Characteristics
✅ Heatmap tập trung ở objects được detect
✅ Boundaries rõ ràng xung quanh objects
✅ Background có attention thấp
✅ Multiple objects đều có coverage

### Bad Visualization Signs
❌ Heatmap lan tỏa khắp ảnh (không specific)
❌ Chỉ highlight một phần nhỏ của object
❌ High attention ở background noise
❌ Không consistent với bounding boxes

## 📚 References

1. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
2. **Grad-CAM++**: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks", WACV 2018
3. **EigenCAM**: Muhammad et al., "Eigen-CAM: Class Activation Map using Principal Components", IJCNN 2020
4. **LayerCAM**: Jiang et al., "LayerCAM: Exploring Hierarchical Class Activation Maps for Localization", TIP 2021

## 🛠️ Troubleshooting Checklist

- [ ] Kiểm tra model path đúng
- [ ] Kiểm tra image path tồn tại
- [ ] Verify layer indices hợp lệ (< số layers của model)
- [ ] Check GPU memory available
- [ ] Clear CUDA cache trước khi chạy
- [ ] Thử giảm số layers nếu OOM
- [ ] Thử conf_threshold khác nếu không có detections

## 💡 Tips & Best Practices

1. **Luôn test với minimal config trước**: `layer=[-3]`
2. **Save output với descriptive names**: Include method, config, confidence
3. **Tạo comparison grid**: Nhiều configs cùng lúc
4. **Document parameters**: Ghi lại config đã dùng cho paper
5. **Use high quality**: Save với quality=95, DPI=300 cho paper
6. **Reproducibility**: Set random seed nếu có randomness

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra error message trong troubleshooting section
2. Verify layer indices với model architecture
3. Test với minimal configuration
4. Check GPU memory available

---

**Version**: 1.0  
**Last Updated**: 2026-01-13  
**Author**: AI Assistant  
**License**: AGPL-3.0

