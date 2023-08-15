import torchvision
import torch
import clip
import r3m
import torchvision.transforms.functional as TF
import geometric_peg_in_hole.models.mae_vit
import torchvision.models
import timm


class ClipResnet50(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model, _ = clip.load('RN50', device)
    
    def forward(self, x):
        x = TF.resize(x, (224, 224))
        x = self.model.encode_image(x)
        return x

class ClipBase(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model, _ = clip.load('ViT-B/16', device)
    
    def forward(self, x):
        x = TF.resize(x, (224, 224))
        x = self.model.encode_image(x)
        return x

def r3m50(frozen, device, output_dim):
    model = r3m.load_r3m('resnet50')
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.to(device)
    return model

def clip50(frozen, device, output_dim):
    model = ClipResnet50(device)
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.to(device)
    return model

def imagenet50(frozen, device, output_dim):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.to(device)
    return model

def imagenet18(frozen, device, output_dim):
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.to(device)
    return model

def resnet18(frozen, device, output_dim):
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Identity()
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.to(device)
    return model

def resnet50(frozen, device, output_dim):
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Identity()
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.to(device)
    return model

def imagenet_base(frozen, device, output_dim):
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.fc_norm = torch.nn.Identity()
    model.head = torch.nn.Identity()
    model.to(device)
    return model

def imagenet_avgpool_base(frozen, device, output_dim):
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.fc_norm = torch.nn.Identity()
    model.head = torch.nn.Identity()
    model.global_pool = True
    model.to(device)
    return model

def mae_base(frozen, device, output_dim):
    model = lib.models.mae_vit.vit_base_patch16()
    delattr(model, 'head')
    model.load_state_dict(torch.load('pretrained/mae_pretrain_vit_base.pth')['model'])
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.fc_norm = torch.nn.Identity()
    model.head = torch.nn.Identity()
    model.to(device)
    return model

def dino_base(frozen, device, output_dim):
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.to(device)
    return model

def dinov2_base(frozen, device, output_dim):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.to(device)
    return model

def clip_base(frozen, device, output_dim):
    model = ClipBase(device)
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.to(device)
    return model
