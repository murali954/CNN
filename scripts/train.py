import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.resnet_yolo import ResNetYOLO
from utils.coco_dataset import CocoYoloDataset  # Assume you implemented this for COCO-format YOLO
from models.utils import non_max_suppression

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetYOLO(num_classes=80).to(device)

    train_dataset = CocoYoloDataset("dataset/train/images", "dataset/train/labels")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()  # Replace with better YOLO loss in real case

    for epoch in range(10):
        model.train()
        total_loss = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)  # Replace with actual YOLO loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "models/model_weights.pth")

if __name__ == "__main__":
    train()
