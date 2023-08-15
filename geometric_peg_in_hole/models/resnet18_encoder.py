import torchvision
import torch

def resnet18(frozen, device, output_dim):
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Identity()
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.to(device)
    return model