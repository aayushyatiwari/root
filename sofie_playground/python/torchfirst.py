import torch
import torch.nn as nn
import ROOT 

model = nn.Sequential(
    torch.nn.Linear(3, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1)
)

torch.onnx.export(model, torch.zeros(1, 3), "models/torchfirst.onnx", dynamo=True)



