import torch
import timm
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mapping taken from training output (print(classes))
IDX_TO_CLASS = {
    0: "Normal",
    1: "MS",
    2: "BT_glioma",
    3: "BT_meningioma",
    4: "AD_VeryMildDemented",
    5: "AD_MildDemented",
    6: "AD_ModerateDemented",
    7: "BT_pituitary"
}

NUM_CLASSES = len(IDX_TO_CLASS)

def load_model():
    model = timm.create_model(
        "rexnet_150",
        pretrained = False,
        num_classes = NUM_CLASSES
    )

    model.load_state_dict(torch.load(
        "saved_model/rexnet_model_best_acc.pth", 
        map_location=DEVICE
    ))
    model.to(DEVICE) 

    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])
    

