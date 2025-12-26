# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from IPython.display import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolov8n.pt",  # any yolov8/yolov9/yolo11/yolo12/rt-detr det model is supported
    # confidence_threshold=0.35,
    # device="cuda",  # or 'cuda:0' if GPU is available
    # imgsz=640,
)


result = get_sliced_prediction(
    "/home/lqc/Research/Detection/ultralytics/0000030_00754_d_0000036.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
    # overlap_height_ratio=0.2,
    # overlap_width_ratio=0.2,
)

# result = get_prediction("/home/lqc/Research/Detection/ultralytics/0000030_00754_d_0000036.jpg", detection_model)
result.export_visuals(export_dir="demo_data/")
Image("demo_data/prediction_visual.png")
# Load a pretrained YOLO11n model
# model = YOLO("yolov8n.pt")

# # Run inference on 'bus.jpg' with arguments
# model.predict("/home/lqc/Research/Detection/ultralytics/0000030_00754_d_0000036.jpg", save=True, imgsz=640, conf=0.5)
