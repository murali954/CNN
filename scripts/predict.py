import torch
from PIL import Image
from torchvision import transforms
from models.resnet_yolo import ResNetYOLO
from models.utils import non_max_suppression

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetYOLO(num_classes=80)
    model.load_state_dict(torch.load("models/model_weights.pth", map_location=device))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        detections = non_max_suppression(output)

    print("Detections:", detections)

if __name__ == "__main__":
    predict("dataset/valid/images/sample.jpg")
