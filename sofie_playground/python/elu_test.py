import torch
import torch.nn as nn
torch.manual_seed(42) # for sanity checking using the same weights
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ELU()
)

model.eval()
m = torch.jit.script(model)
torch.jit.save(m, "elu_model.pt")
print('-'*60)
print("ELU MODEL saved")
print('-'*60)
m = torch.jit.load("elu_model.pt")
m.eval()
x = torch.tensor([[1.0, -1.0, 0.5, -0.5],
                                    [2.0, -2.0, 1.5, -1.5]])
with torch.no_grad():
        out = m(x)
print("Input:\n", x.numpy())
print("Output:\n", out.numpy())
