import torch

from model import BaseANN, BaseCNN


def load():
    model = BaseCNN()
    state_dict = torch.load("checkpoint.pth")
    model.load_state_dict(state_dict)