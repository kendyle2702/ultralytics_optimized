# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg.pt")

# Define visual prompts based on a separate reference image
visual_prompts = dict(
    bboxes=np.array([[221.52, 405.8, 344.98, 857.54]]),  # Box enclosing person
    cls=np.array([0]),  # ID to be assigned for person
)

# Run prediction on a different image, using reference image to guide what to look for
results = model.predict(
    "ultralytics/assets/zidane.jpg",  # Target image for detection
    refer_image="ultralytics/assets/bus.jpg",  # Reference image used to get visual prompts
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
)

# Show results
results[0].show()
