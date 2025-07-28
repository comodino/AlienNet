import torch
from torchvision.datasets import FakeData
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=8, num_classes=10, image_size=(3, 224, 224), dataset_size=100):
    # Trasformazioni di base per immagini RGB
    transform = transforms.Compose([
        transforms.Resize((image_size[1], image_size[2])),
        transforms.ToTensor()
    ])

    # FakeData multilabel: le etichette sono vettori binari di dimensione num_classes
    class MultilabelFakeData(FakeData):
        def __getitem__(self, index):
            img, _ = super().__getitem__(index)
            label = torch.randint(0, 2, (num_classes,), dtype=torch.float32)
            return img, label

    train_dataset = MultilabelFakeData(size=dataset_size, image_size=image_size, transform=transform)
    val_dataset = MultilabelFakeData(size=dataset_size//5, image_size=image_size, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
