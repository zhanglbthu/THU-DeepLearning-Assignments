from torchvision import models
import torch.nn as nn

def model_A(num_classes):
    model_resnet = models.resnet18() 
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    print(model_resnet)
    return model_resnet


def model_B(num_classes):
    # your code here
    pass
