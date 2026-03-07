import matplotlib
matplotlib.use("Agg")
import cv2
import numpy as np
import os
import uuid
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from app.model_loader import model, DEVICE, transform

TARGET_LAYER = model.features[-1] # The last feature of the model is typically the most relevant for Grad-CAM.

def generate_gradcam(image_path, class_idx):
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((224, 224))
    image_np = np.array(image_resized) / 255.0 #for visualization, we need the image in [0, 1] range.
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    if cam is None:
        cam = GradCAMPlusPlus(
            model = model,   # Initialize Grad-CAM with the loaded model.
            target_layers = [TARGET_LAYER],
            use_cuda = (DEVICE == "cuda")
        )
    grayscale_cam = cam(
        input_tensor = input_tensor,
        targets = None # no specific class we want to focus on
    )[0] # Get the CAM for the first (and only) image in the batch.

    visualization = show_cam_on_image(
        image_np,
        grayscale_cam,
        use_rgb = True
    )

    os.makedirs("static/gradcam", exist_ok = True)
    cam_name = f"cam_{uuid.uuid4()}.jpg"
    cam_path = f"static/gradcam/{cam_name}"

    cv2.imwrite(
        cam_path,
        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    )

    return cam_path

