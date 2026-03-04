from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import uuid
import os
import shutil # to copy the uploaded file to the desired location
from app.predict import predict_image
from app.gradcam import generate_gradcam
from app.disorder_info import DISORDER_INFO

app = FastAPI(title = "Neurological Disorder Classification API")

app.mount("/static", StaticFiles(directory = "static"), name = "static")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file extension
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code = 400,
            detail = "Invalid file type. Only .jpg, .jpeg, and .png files are allowed."
        )
    
    # Save the uploaded file to a temporary location
    temp_name = f"temp_{uuid.uuid4()}{ext}"

    try:
        with open(temp_name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        pred_class, class_idx, confidence, all_probs = predict_image(temp_name)
        cam_path = generate_gradcam(temp_name, class_idx)

        return {
            "prediction": pred_class,
            "confidence": confidence,
            "explaination": DISORDER_INFO[pred_class],
            "gradcam_image": f"/{cam_path}",
            "all_probs": all_probs
        }
    
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)






