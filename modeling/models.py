import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)
    

# architecture from https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0283396
encoder_original = lambda nfeat : nn.Sequential(
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
    nn.Conv2d(256, nfeat, 3, stride=2, padding=1)
)

decoder_original = lambda nfeat : nn.Sequential(
    nn.ReLU(True),
    nn.Upsample(scale_factor=2),
    nn.Conv2d(nfeat, 256, (3, 3), bias=False, padding=1),
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

encoder_vq = lambda nfeat: nn.Sequential(
    encoder_expanded,
    
)