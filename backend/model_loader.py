import torch
from model import UNetAutoencoder

def load_model(path: str):
    model = UNetAutoencoder()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

