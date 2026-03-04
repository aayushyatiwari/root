# notes for excercize 4 

**Changes in RModelParser_PyTorch.cxx**
- implemented MakePyTorch_Elu function in the file using reference from MakePyTorch_Selu
- implemented MakePyTorch_MaxPool2D using reference from MakePyTorch_Conv

## What was done for each operator:
1. Wrote MakePyTorch* function in RModelParser_PyTorch.cxx
2. Added declaration at top of file
3. Added entry to mapPyTorchNode with correct ONNX node type string (onnx::Elu, onnx::MaxPool)
4. Added cmath to AddNeededStdLib for ELU (MaxPool already does it internally in ROperator_Pool)

### Pending:
- [x] Proper numerical testing — compare PyTorch output vs SOFIE generated output for ELU and MaxPool2D 
- [] BatchNorm2D, RNN, LSTM, GRU still to implement
