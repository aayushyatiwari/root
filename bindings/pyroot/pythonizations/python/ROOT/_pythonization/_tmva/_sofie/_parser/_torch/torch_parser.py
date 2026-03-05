import os
import time
from .layers.elu import MakeTorchElu

mapTorchLayer = {
    "ELU": MakeTorchElu,
}

def add_layer_into_RModel(rmodel, layer_data):
    from ROOT.TMVA.Experimental import SOFIE
    fLayerType = layer_data["layerType"]
    if fLayerType in mapTorchLayer:
        op = mapTorchLayer[fLayerType](layer_data)
        rmodel.AddOperatorReference(op)
    else:
        raise Exception(f"TMVA.SOFIE - PyTorch layer {fLayerType} is not yet supported")
    return rmodel

class PyTorch:
    def Parse(filename, input_shape, batch_size=1):
        import torch
        import numpy as np
        from ROOT.TMVA.Experimental import SOFIE

        if not os.path.exists(filename):
            raise RuntimeError(f"Model file {filename} not found!")

        model = torch.jit.load(filename)
        model.eval()

        sep = "\\" if os.name == "nt" else "/"
        isep = filename.rfind(sep)
        filename_nodir = filename[isep + 1:] if isep != -1 else filename

        ttime = time.time()
        parsetime = time.asctime(time.gmtime(ttime))
        rmodel = SOFIE.RModel.RModel(filename_nodir, parsetime)

        print(f"PyTorch: parsing model {filename_nodir}")

        layer_iter = 0
        prev_name = ""
        for name, module in model.named_modules():
            if name == "":
                continue
            layer_type = type(module).__name__
            input_name = "input_0" if layer_iter == 0 else f"output_{prev_name}"
            output_name = f"output_{name}"
            layer_data = {
                "layerType":       layer_type,
                "layerAttributes": module.__dict__,
                "layerInput":      [input_name],
                "layerOutput":     [output_name],
                "layerDType":      "float32",
                "layerWeight":     []
            }
            rmodel = add_layer_into_RModel(rmodel, layer_data)
            prev_name = name
            layer_iter += 1

        inputDType = SOFIE.ConvertStringToType("float32")
        inputShape = [batch_size] + list(input_shape)
        rmodel.AddInputTensorInfo("input_0", inputDType, inputShape)
        rmodel.AddInputTensorName("input_0")

        outputNames = [f"output_{prev_name}"]
        rmodel.AddOutputTensorNameList(outputNames)

        return rmodel
