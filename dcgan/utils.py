import os
import torch
from torch import nn

def load_model(model, file_name, device):
    if os.path.exists(file_name):
        model.load_state_dict(torch.load(file_name, map_location=device))
    else:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)
