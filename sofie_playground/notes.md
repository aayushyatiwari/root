# notes after excersize 3 
TMVA_SOFIE_Keras.py exists
TMVA_SOFIE_PyTorch.C exists
TMVA_SOFIE_ONNX.py exists
TMVA_SOFIE_ONNX.C exists

- there is no keras c++ parser and is deprecated so `cpp/TMVA_SOFIE_Keras.C` won't work.
- there is no pytorch python parser 

## Excercise 1 
- [cloned the forked repo](https://github.com/aayushyatiwari/root) 
- git branch 'sofie-dev'

## Excercise 2
- ran all the tutorials that were mentioned.
- tried implementing `TMVA_SOFIE_Keras.C` because it wasn't there and then understood won't work because keras cpp parser is deprecated.

## Exercise 3 

- Understood parser asymmetry — Keras is Python-only, PyTorch is C++-only
- The "missing" file is TMVA_SOFIE_PyTorch.py — which is essentially just TMVA_SOFIE_ONNX.py
- All four parser combinations are covered by existing tutorials
- Played with dummy models 




### notes to myself
> Setup confirmed working:
ROOT 6.39.01 with SOFIE, Python 3.11, torchgpu conda env

> ParseIf.cxx understood:

Reads a boolean condition input
Finds then_branch and else_branch sub-graphs from node attributes
Recursively parses both into RModel objects
Wraps everything into ROperator_If
Registers output types from the then_branch

> Exercise 3 completed:

Understood parser asymmetry — Keras is Python-only, PyTorch is C++-only
The "missing" file is TMVA_SOFIE_PyTorch.py — which is essentially just TMVA_SOFIE_ONNX.py
All four parser combinations are covered by existing tutorials
Played with dummy models in ~/sofie_playground/

Key parse calls learned:

Keras: ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse("model.keras", batch_size=1) — Python only
PyTorch: goes through ONNX, SOFIE::PyTorch::Parse("model.pt", inputShapes) — C++
ONNX: available in both C++ and Python

Next: Exercise 4 — implementing missing ops (ELU, MaxPool2D, BatchNorm2D, RNN, LSTM, GRU) in the parsers.
