import torch
import torch.nn as nn

model = nn.Sequential(
    nn.MaxPool2d(kernel_size=2, stride=2)
)

model.eval()
scripted = torch.jit.script(model)
torch.jit.save(scripted, "models/maxpool2d_model.pt")
