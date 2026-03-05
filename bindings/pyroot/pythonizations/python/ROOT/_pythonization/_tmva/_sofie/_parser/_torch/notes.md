# SOFIE ML4EP — PyTorch Parser (`_torch`)

Python module for loading `.pt` model files and extracting layer information in a structured format, as part of ROOT's SOFIE inference framework.
- Work done as part of an excersise for GSoC

## Structure

```
parser/
├── _keras/
└── _torch/
    ├── extract.py          # standalone extractor, no ROOT dependency
    ├── torch_parser.py     # SOFIE integration, returns RModel
    └── test.py     # test suite
```

## Usage

```python
from extract import extract

result = extract("model.pt")
```

### Example Return format

```python
{
    "model_name": "model.pt",
    "num_layers": 3,
    "layers": [
        {
            "name":       "0",
            "type":       "ELU",
            "attributes": {"alpha": 1.5},
            "weights":    {},
            "input":      "input_0",
            "output":     "output_0",
            "dtype":      "float32"
        },
        ...
    ]
}
```

## Supported Layers

| Layer | Attributes | Weights |
|---|---|---|
| `ELU` | `alpha` | none |
| `MaxPool2d` | `kernel_size`, `stride`, `padding`, `dilation` | none |
| `BatchNorm2d` | `eps`, `momentum` | `weight`, `bias`, `running_mean`, `running_var` |
| `RNN` | `input_size`, `hidden_size`, `num_layers`, `nonlinearity` | `weight_ih_lN`, `weight_hh_lN`, `bias_ih_lN`, `bias_hh_lN` |
| `LSTM` | `input_size`, `hidden_size`, `num_layers` | same as RNN, shaped for 4 gates |
| `GRU` | `input_size`, `hidden_size`, `num_layers` | same as RNN, shaped for 3 gates |

Stacked recurrent layers (`num_layers > 1`) are supported — weights are extracted per layer as `weight_ih_l0`, `weight_ih_l1`, etc.

## Running Tests

```bash
python test.py
```

## Future Prospects

### `torch_parser.py`
`extract()` is the foundation for `torch_parser.py`, which takes the extracted layer data and builds a SOFIE `RModel` object — the same format produced by the existing `PyKeras` parser. This enables end-to-end PyTorch inference code generation in C++ via ROOT's SOFIE framework.

```python
from torch_parser import PyTorch
rmodel = PyTorch.Parse("model.pt", input_shape=[3, 224, 224])
rmodel.Generate()
```

### Planned additions
- More layer types: `Conv2d`, `Linear`, `Dropout`, `LayerNorm`
- Bidirectional RNN/LSTM/GRU support
- Multi-input and skip-connection model support
- Numerical validation against PyTorch inference output (high priority)
