import matplotlib
matplotlib.use("Agg")

import cv2
import numpy as np
import os
import uuid
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from app.model_loader import model, DEVICE, transform


TARGET_LAYER = model.features[-1]

# Lazy initialization
cam = None


def generate_gradcam(image_path, class_idx):
    global cam

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((224, 224))
    image_np = np.array(image_resized) / 255.0

    # Prepare tensor
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Initialize GradCAM once
    if cam is None:
        cam = GradCAM(
            model=model,
            target_layers=[TARGET_LAYER],
            use_cuda=(DEVICE == "cuda")
        )

    # Focus on predicted class
    targets = [ClassifierOutputTarget(class_idx)]

    # Generate CAM
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # Overlay heatmap
    visualization = show_cam_on_image(
        image_np,
        grayscale_cam,
        use_rgb=True
    )

    # Save image
    os.makedirs("static/gradcam", exist_ok=True)

    cam_name = f"cam_{uuid.uuid4()}.jpg"
    cam_path = f"static/gradcam/{cam_name}"

    cv2.imwrite(
        cam_path,
        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    )

    return cam_path