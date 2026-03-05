import os
import torch
import torch.nn as nn
from extract import extract


def test_elu():
    model = nn.Sequential(nn.ELU(alpha=1.5))
    torch.jit.save(torch.jit.script(model), "test_elu.pt")
    result = extract("test_elu.pt")

    assert result["num_layers"] == 1
    assert result["model_name"] == "test_elu.pt"
    layer = result["layers"][0]
    assert layer["type"] == "ELU"
    assert layer["attributes"]["alpha"] == 1.5
    assert layer["input"] == "input_0"
    assert layer["output"] == "output_0"
    assert layer["weights"] == {}

    os.remove("test_elu.pt")
    print("test_elu passed")


def test_maxpool():
    model = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
    torch.jit.save(torch.jit.script(model), "test_maxpool.pt")
    result = extract("test_maxpool.pt")

    assert result["num_layers"] == 1
    assert result["model_name"] == "test_maxpool.pt"
    layer = result["layers"][0]
    assert layer["type"] == "MaxPool2d"
    assert layer["input"] == "input_0"
    assert layer["output"] == "output_0"
    assert layer["weights"] == {}

    os.remove("test_maxpool.pt")
    print("test_maxpool passed")


def test_batchnorm():
    model = nn.Sequential(nn.BatchNorm2d(16))
    torch.jit.save(torch.jit.script(model), "test_batchnorm.pt")
    result = extract("test_batchnorm.pt")

    assert result["num_layers"] == 1
    assert result["model_name"] == "test_batchnorm.pt"
    layer = result["layers"][0]
    assert layer["type"] == "BatchNorm2d"
    assert layer["attributes"]["eps"] == 1e-05
    assert layer["attributes"]["momentum"] == 0.1
    assert layer["input"] == "input_0"
    assert layer["output"] == "output_0"
    assert layer["weights"]["weight"].shape == (16,)
    assert layer["weights"]["bias"].shape == (16,)
    assert layer["weights"]["running_mean"].shape == (16,)
    assert layer["weights"]["running_var"].shape == (16,)
    assert torch.all(layer["weights"]["weight"] == 1.)
    assert torch.all(layer["weights"]["bias"] == 0.)
    assert torch.all(layer["weights"]["running_mean"] == 0.)
    assert torch.all(layer["weights"]["running_var"] == 1.)

    os.remove("test_batchnorm.pt")
    print("test_batchnorm passed")


def test_rnn():
    model = nn.RNN(input_size=10, hidden_size=20, num_layers=1)
    torch.jit.save(torch.jit.script(model), "test_rnn.pt")
    result = extract("test_rnn.pt")

    assert result["num_layers"] == 1
    layer = result["layers"][0]
    assert layer["type"] == "RNN"
    assert layer["attributes"]["input_size"] == 10
    assert layer["attributes"]["hidden_size"] == 20
    assert layer["attributes"]["num_layers"] == 1
    assert layer["attributes"]["nonlinearity"] == "tanh"
    assert layer["input"] == "input_0"
    assert layer["output"] == "output_0"
    assert layer["weights"]["weight_ih_l0"].shape == (20, 10)
    assert layer["weights"]["weight_hh_l0"].shape == (20, 20)
    assert layer["weights"]["bias_ih_l0"].shape == (20,)
    assert layer["weights"]["bias_hh_l0"].shape == (20,)

    os.remove("test_rnn.pt")
    print("test_rnn passed")


def test_rnn_stacked():
    model = nn.RNN(input_size=10, hidden_size=20, num_layers=2)
    torch.jit.save(torch.jit.script(model), "test_rnn_stacked.pt")
    result = extract("test_rnn_stacked.pt")

    layer = result["layers"][0]
    assert layer["type"] == "RNN"
    assert layer["attributes"]["num_layers"] == 2
    assert layer["weights"]["weight_ih_l0"].shape == (20, 10)
    assert layer["weights"]["weight_hh_l0"].shape == (20, 20)
    assert layer["weights"]["weight_ih_l1"].shape == (20, 20)
    assert layer["weights"]["weight_hh_l1"].shape == (20, 20)
    assert layer["weights"]["bias_ih_l0"].shape == (20,)
    assert layer["weights"]["bias_hh_l0"].shape == (20,)
    assert layer["weights"]["bias_ih_l1"].shape == (20,)
    assert layer["weights"]["bias_hh_l1"].shape == (20,)

    os.remove("test_rnn_stacked.pt")
    print("test_rnn_stacked passed")


def test_lstm():
    model = nn.LSTM(input_size=10, hidden_size=20, num_layers=1)
    torch.jit.save(torch.jit.script(model), "test_lstm.pt")
    result = extract("test_lstm.pt")

    assert result["num_layers"] == 1
    layer = result["layers"][0]
    assert layer["type"] == "LSTM"
    assert layer["attributes"]["input_size"] == 10
    assert layer["attributes"]["hidden_size"] == 20
    assert layer["attributes"]["num_layers"] == 1
    assert layer["input"] == "input_0"
    assert layer["output"] == "output_0"
    assert layer["weights"]["weight_ih_l0"].shape == (80, 10)
    assert layer["weights"]["weight_hh_l0"].shape == (80, 20)
    assert layer["weights"]["bias_ih_l0"].shape == (80,)
    assert layer["weights"]["bias_hh_l0"].shape == (80,)

    os.remove("test_lstm.pt")
    print("test_lstm passed")


def test_gru():
    model = nn.GRU(input_size=10, hidden_size=20, num_layers=1)
    torch.jit.save(torch.jit.script(model), "test_gru.pt")
    result = extract("test_gru.pt")

    assert result["num_layers"] == 1
    layer = result["layers"][0]
    assert layer["type"] == "GRU"
    assert layer["attributes"]["input_size"] == 10
    assert layer["attributes"]["hidden_size"] == 20
    assert layer["attributes"]["num_layers"] == 1
    assert layer["input"] == "input_0"
    assert layer["output"] == "output_0"
    assert layer["weights"]["weight_ih_l0"].shape == (60, 10)
    assert layer["weights"]["weight_hh_l0"].shape == (60, 20)
    assert layer["weights"]["bias_ih_l0"].shape == (60,)
    assert layer["weights"]["bias_hh_l0"].shape == (60,)

    os.remove("test_gru.pt")
    print("test_gru passed")


def test_multi_layer():
    model = nn.Sequential(
        nn.ELU(alpha=1.5),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    torch.jit.save(torch.jit.script(model), "test_multi.pt")
    result = extract("test_multi.pt")

    assert result["num_layers"] == 3
    assert result["model_name"] == "test_multi.pt"
    layers = result["layers"]
    assert layers[0]["type"] == "ELU"
    assert layers[1]["type"] == "BatchNorm2d"
    assert layers[2]["type"] == "MaxPool2d"
    assert layers[0]["input"] == "input_0"
    assert layers[1]["input"] == layers[0]["output"]
    assert layers[2]["input"] == layers[1]["output"]

    os.remove("test_multi.pt")
    print("test_multi_layer passed")


if __name__ == "__main__":
    tests = [
        test_elu,
        test_maxpool,
        test_batchnorm,
        test_rnn,
        test_rnn_stacked,
        test_lstm,
        test_gru,
        test_multi_layer,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"{test.__name__} FAILED: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
