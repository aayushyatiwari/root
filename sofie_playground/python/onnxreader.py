import ROOT
import os  
def ParseModel(modelFile, verbose=False):

    parser = ROOT.TMVA.Experimental.SOFIE.RModelParser_ONNX()
    model = parser.Parse(modelFile,verbose)

    #print model weights
    if (verbose):
        model.PrintInitializedTensors()
        data = model.GetTensorData['float']('0weight')
        print("0weight",data)
        data = model.GetTensorData['float']('2weight')
        print("2weight",data)

    # Generating inference code
    model.Generate()
    #generate header file (and .dat file) with modelName+.hxx
    model.OutputGenerated()
    if (verbose) :
        model.PrintGenerated()
     
    modelCode = modelFile.replace(".onnx",".hxx")
    print("Generated model header file ",modelCode)
    return modelCode

if __name__ == "__main__":
    modelname = input("File Name (path): ")
    returnedModel = ParseModel(modelname, verbose=False)
    print(type(returnedModel))
