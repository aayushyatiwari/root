import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ELU()
)

model.eval()
m = torch.jit.script(model)
torch.jit.save(m, "models/elu_model.pt")
print("saved")
