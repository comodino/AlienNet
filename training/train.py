import torch
from models.efficientnet import get_model
from training.dataloader import get_dataloaders

def run_training():
    print("🚀 Inizio addestramento...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilizzo il dispositivo: {device}")

    model = get_model(num_classes=10).to(device)
    train_loader, val_loader = get_dataloaders()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "experiments/checkpoints/efficientnet.pth")
    print("✅ Addestramento completato e modello salvato.")

if __name__ == "__main__":
    run_training()
