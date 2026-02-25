import torch
from PIL import Image
from app.model_loader import model, transform, IDX_TO_CLASS, DEVICE

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim = 1) # Apply softmax to get probabilities. 
        conf, pred_idx = torch.max(probs, dim = 1)

    predicted_class = IDX_TO_CLASS[pred_idx.item()]
    confidence = round(conf.item() * 100, 2)
    return predicted_class, pred_idx.item(), confidence 