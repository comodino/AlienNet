import torch
from efficientnet_pytorch import EfficientNet

def get_model(num_classes):
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = torch.nn.Linear(model._fc.in_features, num_classes)
    return model
