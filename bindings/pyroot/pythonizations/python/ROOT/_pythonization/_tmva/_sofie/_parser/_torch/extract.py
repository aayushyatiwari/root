import os
import torch


def _extract_elu(module):
    return {
        "attributes": {"alpha": module.alpha},
        "weights": {}
    }


def _extract_maxpool2d(module):
    return {
        "attributes": {
            "kernel_size": module.kernel_size,
            "stride": module.stride,
            "padding": module.padding,
            "dilation": module.dilation
        },
        "weights": {}
    }


def _extract_batchnorm2d(module):
    return {
        "attributes": {
            "eps": module.eps,
            "momentum": module.momentum
        },
        "weights": {
            "weight": module.weight.data,
            "bias": module.bias.data,
            "running_mean": module.running_mean,
            "running_var": module.running_var
        }
    }


def _extract_rnn(module):
    weights = {}
    for k in range(module.num_layers):
        weights[f"weight_ih_l{k}"] = getattr(module, f"weight_ih_l{k}").data
        weights[f"weight_hh_l{k}"] = getattr(module, f"weight_hh_l{k}").data
        weights[f"bias_ih_l{k}"] = getattr(module, f"bias_ih_l{k}").data
        weights[f"bias_hh_l{k}"] = getattr(module, f"bias_hh_l{k}").data
    return {
        "attributes": {
            "input_size": module.input_size,
            "hidden_size": module.hidden_size,
            "num_layers": module.num_layers,
            "nonlinearity": module.nonlinearity
        },
        "weights": weights
    }


def _extract_lstm(module):
    weights = {}
    for k in range(module.num_layers):
        weights[f"weight_ih_l{k}"] = getattr(module, f"weight_ih_l{k}").data
        weights[f"weight_hh_l{k}"] = getattr(module, f"weight_hh_l{k}").data
        weights[f"bias_ih_l{k}"] = getattr(module, f"bias_ih_l{k}").data
        weights[f"bias_hh_l{k}"] = getattr(module, f"bias_hh_l{k}").data
    return {
        "attributes": {
            "input_size": module.input_size,
            "hidden_size": module.hidden_size,
            "num_layers": module.num_layers
        },
        "weights": weights
    }


def _extract_gru(module):
    weights = {}
    for k in range(module.num_layers):
        weights[f"weight_ih_l{k}"] = getattr(module, f"weight_ih_l{k}").data
        weights[f"weight_hh_l{k}"] = getattr(module, f"weight_hh_l{k}").data
        weights[f"bias_ih_l{k}"] = getattr(module, f"bias_ih_l{k}").data
        weights[f"bias_hh_l{k}"] = getattr(module, f"bias_hh_l{k}").data
    return {
        "attributes": {
            "input_size": module.input_size,
            "hidden_size": module.hidden_size,
            "num_layers": module.num_layers
        },
        "weights": weights
    }


_extractor_map = {
    "ELU":          _extract_elu,
    "MaxPool2d":    _extract_maxpool2d,
    "BatchNorm2d":  _extract_batchnorm2d,
    "RNN":          _extract_rnn,
    "LSTM":         _extract_lstm,
    "GRU":          _extract_gru
}


def extract(filename):
    if not os.path.exists(filename):
        raise RuntimeError(f"Model file {filename} not found!")

    model = torch.jit.load(filename)
    model.eval()

    layers = []
    layer_iter = 0
    prev_output = "input_0"

    for name, module in model.named_modules():
        if name == "": # skip for standalone Sequential
            layer_type = getattr(module, "original_name",type(module).__name__ )
            if layer_type in _extractor_map:
                name = '0'
            else:
                continue

        layer_type = getattr(module, 'original_name', type(module).__name__)
        print(layer_type)
        if layer_type not in _extractor_map:
            raise Exception(f"TMVA.SOFIE - PyTorch layer {layer_type} is not yet supported")

        input_name = prev_output
        output_name = f"output_{name}"

        extracted = _extractor_map[layer_type](module)

        layer_data = {
            "name":       name,
            "type":       layer_type,
            "attributes": extracted["attributes"],
            "weights":    extracted["weights"],
            "input":      input_name,
            "output":     output_name,
            "dtype":      "float32"
        }

        layers.append(layer_data)
        prev_output = output_name
        layer_iter += 1

    return {
        "model_name": os.path.basename(filename),
        "num_layers": layer_iter,
        "layers":     layers
    }
