import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torch

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)

encoder_expanded = lambda nfeat : nn.Sequential(
    nn.Conv2d(1, 32, 3, stride=2, bias=False, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    nn.Conv2d(32, 64, 3, stride=2, bias=False, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.Conv2d(64, 128, 3, stride=2, bias=False, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.Conv2d(128, 256, 3, stride=2, bias=False, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.Conv2d(256, 512, 3, stride=2, bias=False, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.Conv2d(512, nfeat, 3, stride=2, padding=1),
    
    Reshape(nfeat * 4 * 4)
)

decoder_expanded = lambda nfeat : nn.Sequential(
    Reshape(nfeat, 4, 4),
    nn.ReLU(True),
    
    nn.Upsample(scale_factor=2),
    nn.Conv2d(nfeat, 512, (3, 3), bias=False, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.Conv2d(512, 512, (3, 3), bias=False, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    
    nn.Upsample(scale_factor=2),
    nn.Conv2d(512, 256, (3, 3), bias=False, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.Conv2d(256, 256, (3, 3), bias=False, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(True),

    nn.Upsample(scale_factor=2),
    nn.Conv2d(256, 128, (3, 3), bias=False, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.Conv2d(128, 128, (3, 3), bias=False, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    nn.Upsample(scale_factor=2),
    nn.Conv2d(128, 64, (3, 3), bias=False, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.Conv2d(64, 64, (3, 3), bias=False, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.Upsample(scale_factor=2),
    nn.Conv2d(64, 32, (3, 3), bias=False, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(True),

    nn.Upsample(scale_factor=2),
    nn.Conv2d(32, 1, (3, 3), bias=False, padding=1),
    nn.BatchNorm2d(1),
    nn.ReLU(True),
    nn.Conv2d(1, 1, (3, 3), bias=False, padding=1)
)

class AutoEncoder():
    
    def __init__(self, num_dims, model_state, device):
        super(AutoEncoder, self).__init__()
        
        self.device = device
        
        self.encoder = encoder_expanded(num_dims)
        self.decoder = decoder_expanded(num_dims)

        self.model = nn.Sequential(
            self.encoder,
            self.decoder
        ).to(device)
        
        state = torch.load(model_state)
        
        self.model.load_state_dict(state)
        
    def embed(self, x):
        label = x.to(self.device)[..., :256]
        
        x = self.encoder(label)
        
        return x.detach().cpu().numpy().squeeze()
        