from torchvision import models
import torch.nn as nn

def model_A(num_classes):
    model_resnet = models.resnet18()
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    print(model_resnet)
    return model_resnet


def model_B(num_classes):
    # DenseNet
    model_densenet = models.densenet121()
    
    # Replace classifier (fully connected) layer
    num_features = model_densenet.classifier.in_features
    model_densenet.classifier = nn.Linear(num_features, num_classes)
    print(model_densenet)
    
    return model_densenet
