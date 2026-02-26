# api/predict.py
import json
from pathlib import Path
import torch
from model import SoilNet
from preprocess import preprocess_image

# Load model once
MODEL_PATH = Path(__file__).parent.parent / "model" / "soil_model.pth"
model = SoilNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

def handler(request):
    try:
        data = json.loads(request.body)
        image_data = data["image"]
        input_tensor = preprocess_image(image_data)
        with torch.no_grad():
            output = model(input_tensor)
        prediction = int(output.argmax(dim=1).item())
        return {
            "status": 200,
            "body": json.dumps({"prediction": prediction})
        }
    except Exception as e:
        return {
            "status": 500,
            "body": json.dumps({"error": str(e)})
        }
