import torch
import os
from models.efficientnet import get_model
from training.dataloader import get_dataloaders
from torch.nn.functional import binary_cross_entropy_with_logits
from pathlib import Path

# Imposta la working directory alla root del progetto
BASE_DIR = Path(__file__).resolve().parent.parent
os.chdir(BASE_DIR)

def run_evaluation():
    print("🔍 Valutazione modello salvato...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(num_classes=10).to(device)

    checkpoint_path = "experiments/checkpoints/efficientnet.pth"
    if not os.path.exists(checkpoint_path):
        print("❌ Checkpoint non trovato:", checkpoint_path)
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    _, val_loader = get_dataloaders()

    total_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = binary_cross_entropy_with_logits(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"📊 Val Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    run_evaluation()
