import torch
import torch.nn as nn
torch.manual_seed(42)
model = nn.Sequential(
    nn.MaxPool2d(kernel_size=2, stride=2)
)

model.eval()
scripted = torch.jit.script(model)
torch.jit.save(scripted, "maxpool2d_model.pt")
print('-'*60)
print('maxpool2d model saved')
print('-'*60)
# loading the model to check Output
scripted = torch.jit.load("maxpool2d_model.pt")
scripted.eval()
x = torch.tensor([[[[1.0, 3.0, 2.0, 4.0],
                                      [5.0, 7.0, 6.0, 8.0],
                                      [9.0, 11.0, 10.0, 12.0],
                                      [13.0, 15.0, 14.0, 16.0]]]])
with torch.no_grad():
        out = scripted(x)
print("Input:\n", x.numpy())
print("Output:\n", out.numpy())
