import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torchvision import models  # type: ignore


def get_model(num_classes=5):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, num_classes),
        nn.Sigmoid()  # Sigmoid for multi-label classification
    )
    return model


def load_model(num_classes=5, model_path=None):
    model = get_model(num_classes)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    return model
