# YOLOv12-Visdrone: Kiến trúc tối ưu cho phát hiện đối tượng nhỏ

## Tổng quan

Đây là 3 kiến trúc YOLOv12 được thiết kế đặc biệt để phát hiện các đối tượng nhỏ trên dataset **Visdrone**. Mỗi kiến trúc có điểm mạnh riêng và được tối ưu cho các trường hợp sử dụng khác nhau.

---

## 📊 So sánh các kiến trúc

| Kiến trúc                      | Độ phức tạp | Tốc độ   | Độ chính xác | Phù hợp cho                       |
| ------------------------------ | ----------- | -------- | ------------ | --------------------------------- |
| **yolo12-visdrone.yaml**       | ⭐⭐⭐      | ⚡⚡⚡⚡ | 📈📈📈       | Cân bằng tốc độ & độ chính xác    |
| **yolo12-visdrone-bifpn.yaml** | ⭐⭐⭐⭐    | ⚡⚡⚡   | 📈📈📈📈     | Độ chính xác cao, nhiều tính năng |
| **yolo12-visdrone-dense.yaml** | ⭐⭐⭐⭐⭐  | ⚡⚡     | 📈📈📈📈📈   | Đối tượng cực nhỏ, nghiên cứu     |

---

## 🎯 Kiến trúc 1: yolo12-visdrone.yaml

### Đặc điểm chính:

- ✅ **P2 Detection Head** (stride 4) - phát hiện đối tượng nhỏ
- ✅ **CBAM Attention** trên P2 và P3 layers
- ✅ **Enhanced FPN** với top-down và bottom-up pathways
- ✅ **4 detection scales**: P2/4, P3/8, P4/16, P5/32

### Ưu điểm:

- Kiến trúc đơn giản, dễ train
- Tốc độ inference nhanh
- Hiệu quả với đối tượng nhỏ-trung bình
- CBAM giúp tập trung vào vùng quan trọng

### Khi nào sử dụng:

- ✅ Triển khai production với yêu cầu real-time
- ✅ GPU/thiết bị có tài nguyên hạn chế
- ✅ Cần cân bằng giữa tốc độ và độ chính xác
- ✅ Đối tượng nhỏ không quá extreme (> 16x16 pixels)

### Training command:

```bash
# Training từ scratch
yolo detect train data=VisDrone.yaml model=ultralytics/cfg/models/12/yolo12-visdrone.yaml epochs=300 imgsz=1024 batch=16

# Fine-tuning từ yolo12 pretrained
yolo detect train data=VisDrone.yaml model=yolo12m.pt cfg=ultralytics/cfg/models/12/yolo12-visdrone.yaml epochs=200 imgsz=1024
```

---

## 🚀 Kiến trúc 2: yolo12-visdrone-bifpn.yaml (KHUYẾN NGHỊ)

### Đặc điểm chính:

- ✅ **BiFPN-inspired connections** - kết nối đa chiều
- ✅ **C2fPSA** (Polarized Self-Attention) - attention mạnh mẽ
- ✅ **Multi-source fusion**: Kết hợp features từ nhiều nguồn
- ✅ **SPPF** (Spatial Pyramid Pooling Fast) - global context
- ✅ **Enhanced P2 pathway** - 3 repeats thay vì 2

### Ưu điểm:

- BiFPN connections cho phép information flow tốt hơn
- PSA attention hiệu quả hơn CBAM cho small objects
- Multi-source fusion tận dụng features ở nhiều levels
- SPPF giúp nắm bắt multi-scale context

### Khi nào sử dụng:

- ✅ **KHUYẾN NGHỊ cho Visdrone**
- ✅ Muốn độ chính xác cao nhất có thể
- ✅ GPU đủ mạnh (RTX 3090, 4090, V100, A100)
- ✅ Có thể chấp nhận tốc độ chậm hơn 20-30%
- ✅ Competition, research, hoặc offline inference

### Training command:

```bash
# Training với scale 's' - khuyến nghị
yolo detect train data=VisDrone.yaml model=ultralytics/cfg/models/12/yolo12-visdrone-bifpn.yaml \
  epochs=300 imgsz=1024 batch=12 scale=s

# Training với scale 'm' - độ chính xác cao
yolo detect train data=VisDrone.yaml model=ultralytics/cfg/models/12/yolo12-visdrone-bifpn.yaml \
  epochs=300 imgsz=1024 batch=8 scale=m optimizer=AdamW lr0=0.001
```

---

## 💎 Kiến trúc 3: yolo12-visdrone-dense.yaml

### Đặc điểm chính:

- ✅ **Dense connections** - DenseNet-inspired
- ✅ **Cross-scale direct connections** - P2 -> P4 trực tiếp
- ✅ **Dual attention** - Channel + Spatial attention riêng biệt
- ✅ **Maximum feature reuse** - tận dụng tối đa features
- ✅ **Stride-4 convolution** - P2 to P4 direct path (SÁNG TẠO!)

### Ưu điểm:

- Hiệu quả nhất với đối tượng cực nhỏ (< 16x16 pixels)
- Dense connections giảm vanishing gradient
- Cross-scale direct paths giữ chi tiết từ resolution cao
- Dual attention mechanism mạnh mẽ

### Nhược điểm:

- Memory consumption cao nhất
- Tốc độ chậm nhất
- Cần GPU memory lớn
- Khó train hơn (dễ overfit)

### Khi nào sử dụng:

- ✅ Đối tượng cực nhỏ, rất khó phát hiện
- ✅ Dataset có nhiều birds, distant pedestrians
- ✅ Nghiên cứu, paper, competition
- ✅ GPU memory >= 24GB
- ✅ Không quan tâm tốc độ inference
- ✅ Muốn push giới hạn của small object detection

### Training command:

```bash
# Training với mixed precision để tiết kiệm memory
yolo detect train data=VisDrone.yaml model=ultralytics/cfg/models/12/yolo12-visdrone-dense.yaml \
  epochs=300 imgsz=1024 batch=6 scale=s amp=True patience=50

# Training với scale 'n' nếu GPU memory hạn chế
yolo detect train data=VisDrone.yaml model=ultralytics/cfg/models/12/yolo12-visdrone-dense.yaml \
  epochs=300 imgsz=896 batch=8 scale=n
```

---

## 🎨 Các đổi mới sáng tạo

### 1. **P2 Detection Head**

```yaml
# Tất cả 3 kiến trúc đều có P2 head
- P2 (stride 4): Phát hiện đối tượng 16-32 pixels
- P3 (stride 8): Phát hiện đối tượng 32-64 pixels
- P4 (stride 16): Phát hiện đối tượng 64-128 pixels
- P5 (stride 32): Phát hiện đối tượng > 128 pixels
```

### 2. **Multi-source Feature Fusion** (BiFPN architecture)

```yaml
# Kết hợp features từ nhiều nguồn
- [[-1, 15, 4], 1, Concat, [1]] # P3_topdown + P3_bottomup + P3_backbone
```

### 3. **Cross-scale Direct Connections** (Dense architecture)

```yaml
# P2 -> P4 trực tiếp (stride 4) - giữ thông tin chi tiết
- [20, 1, Conv, [128, 3, 4]] # Innovative: skip P3, go directly to P4
```

### 4. **Attention Mechanisms**

- **CBAM**: Channel + Spatial attention (đơn giản, hiệu quả)
- **PSA**: Polarized Self-Attention (mạnh hơn, phức tạp hơn)
- **C2fAttn**: Attention-enhanced C2f blocks
- **Dual Attention**: Channel và Spatial riêng biệt

---

## 📋 Hyperparameters khuyến nghị cho Visdrone

### Augmentation cho small objects:

```yaml
# Create visdrone_aug.yaml
hsv_h: 0.015 # Giảm để tránh mất object nhỏ
hsv_s: 0.5
hsv_v: 0.4
degrees: 0 # Không rotate - objects nhỏ dễ mất
translate: 0.1 # Giảm translation
scale: 0.2 # Giảm scale để giữ small objects
shear: 0 # Không shear
perspective: 0 # Không perspective
flipud: 0.0 # Không flip vertical (drones luôn nhìn từ trên)
fliplr: 0.5 # Horizontal flip OK
mosaic: 1.0 # Mosaic tốt cho small objects
mixup: 0.1 # Ít mixup
copy_paste: 0.0 # Không copy-paste (confuse small objects)
```

### Training hyperparameters:

```yaml
# Recommended settings
imgsz: 1024 # Hoặc 1280 nếu GPU đủ mạnh
batch: 8-16 # Tùy GPU
epochs: 300
patience: 50
optimizer: AdamW # Tốt hơn SGD cho small objects
lr0: 0.001 # Learning rate ban đầu
lrf: 0.01 # Final learning rate
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 5
warmup_momentum: 0.8
box: 7.5 # Box loss gain
cls: 0.5 # Class loss gain (giảm vì Visdrone imbalanced)
dfl: 1.5 # DFL loss gain
```

---

## 🔧 Tips & Tricks cho Visdrone

### 1. Image Size

- **Khuyến nghị**: 1024x1024 hoặc 1280x1280
- Visdrone có nhiều objects cực nhỏ, cần resolution cao
- Nếu GPU memory hạn chế, tối thiểu 896x896

### 2. Multi-scale Training & Testing

```bash
# Training với multi-scale
yolo detect train data=VisDrone.yaml model=yolo12-visdrone-bifpn.yaml imgsz=1024 scale=0.5

# Testing với multi-scale TTA (Test Time Augmentation)
yolo detect val model=runs/detect/train/weights/best.pt data=VisDrone.yaml imgsz=1280 augment=True
```

### 3. Confidence & IoU Thresholds

```bash
# Small objects cần thresholds thấp hơn
yolo detect val model=best.pt conf=0.001 iou=0.3 max_det=1000
```

### 4. NMS (Non-Maximum Suppression)

```bash
# Visdrone có nhiều objects chồng chéo
yolo detect val model=best.pt iou=0.3 agnostic_nms=True
```

### 5. Class Imbalance

Visdrone có class imbalance nghiêm trọng:

- **Nhiều**: car, people, van
- **Ít**: tricycle, awning-tricycle, bus

Giải pháp:

```python
# Trong training code, thêm class weights
# hoặc sử dụng focal loss
```

---

## 📈 Kết quả kỳ vọng trên Visdrone

### Với yolo12-visdrone-bifpn.yaml (scale 'm'):

```
Expected mAP50-95: 0.28-0.32
Expected mAP50: 0.48-0.52

Per class (rough estimates):
- Pedestrian: AP50 ~0.42
- People: AP50 ~0.38
- Bicycle: AP50 ~0.28
- Car: AP50 ~0.68
- Van: AP50 ~0.55
- Truck: AP50 ~0.52
- Tricycle: AP50 ~0.32
- Awning-tricycle: AP50 ~0.25
- Bus: AP50 ~0.58
- Motor: AP50 ~0.48
```

---

## 🚀 Quick Start

### 1. Chuẩn bị dataset Visdrone

```bash
# Download Visdrone dataset
# Organize theo format YOLO
dataset/
├── train/
│ ├── images/
│ └── labels/
├── val/
│ ├── images/
│ └── labels/
└── VisDrone.yaml
```

### 2. Chọn kiến trúc

- **Production/Real-time**: yolo12-visdrone.yaml
- **Cân bằng tốt nhất**: yolo12-visdrone-bifpn.yaml ⭐
- **Maximum accuracy**: yolo12-visdrone-dense.yaml

### 3. Training

```bash
# Ví dụ với BiFPN (khuyến nghị)
yolo detect train \
  data=VisDrone.yaml \
  model=ultralytics/cfg/models/12/yolo12-visdrone-bifpn.yaml \
  epochs=300 \
  imgsz=1024 \
  batch=12 \
  device=0 \
  project=visdrone \
  name=yolo12-bifpn-m \
  scale=m
```

### 4. Validation

```bash
yolo detect val \
  model=visdrone/yolo12-bifpn-m/weights/best.pt \
  data=VisDrone.yaml \
  imgsz=1024 \
  conf=0.001 \
  iou=0.3 \
  max_det=1000
```

### 5. Inference

```bash
yolo detect predict \
  model=visdrone/yolo12-bifpn-m/weights/best.pt \
  source=test_images/ \
  imgsz=1024 \
  conf=0.25 \
  save=True \
  save_txt=True
```

---

## 🔬 Experiments & Ablation Studies

### Thử nghiệm các variants:

1. **Với/Không P2 head**

   ```bash
   # So sánh mAP của small objects
   ```

2. **Attention mechanisms**
   - CBAM vs PSA vs None
   - Dual attention (Channel + Spatial) vs Single

3. **FPN variants**
   - Standard FPN
   - BiFPN
   - Dense FPN

4. **Backbone depth**
   - C3k2 repeats: 2 vs 3 vs 4

---

## 📚 References & Inspiration

1. **BiFPN**: "EfficientDet: Scalable and Efficient Object Detection"
2. **PSA**: "Polarized Self-Attention: Towards High-quality Pixel-wise Regression"
3. **CBAM**: "Convolutional Block Attention Module"
4. **Dense Connections**: "Densely Connected Convolutional Networks"
5. **Feature Pyramid**: "Feature Pyramid Networks for Object Detection"

---

## 🤝 Contributing

Nếu bạn có ý tưởng cải tiến hoặc kết quả tốt hơn, hãy chia sẻ!

---

## ⚠️ Lưu ý quan trọng

1. **GPU Memory**: Kiến trúc dense cần GPU >= 24GB
2. **Training time**: Với Visdrone, cần 300+ epochs để converge
3. **Patience**: Small object detection khó, cần kiên nhẫn tune hyperparameters
4. **Validation**: Luôn validate trên test set riêng, không overfit trên val set
5. **Class imbalance**: Xem xét weighted loss hoặc focal loss

---

**Chúc bạn training thành công! 🚀**

Made with ❤️ for Visdrone small object detection
