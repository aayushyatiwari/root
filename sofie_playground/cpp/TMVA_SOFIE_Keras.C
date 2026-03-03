/// \file
/// \ingroup tutorial_ml
/// \notebook -nodraw
/// This macro provides a simple example for the parsing of Keras .h5 file
/// into RModel object and further generating the .hxx header files for inference.
///
/// \macro_code
/// \macro_output
/// \author Aayushya

using namespace TMVA::Experimental;

TString pythonSrc = "\
import tensorflow as tf\n\
import numpy as np\n\
\n\
model = tf.keras.Sequential([\n\
    tf.keras.layers.Dense(16, activation='relu', input_shape=(32,)),\n\
    tf.keras.layers.Dense(8)\n\
])\n\
\n\
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mse')\n\
\n\
x = np.random.randn(2, 32)\n\
y = np.random.randn(2, 8)\n\
\n\
model.fit(x, y, epochs=500, verbose=0)\n\
model.save('models/KerasModel.keras')\n";

void TMVA_SOFIE_Keras(){
   TMacro m;
   m.AddLine(pythonSrc);
   m.SaveSource("make_keras_model.py");
   gSystem->Exec("python3 make_keras_model.py");

   SOFIE::RModel model = SOFIE::PyKeras::Parse("models/KerasModel.keras");

   model.Generate();
   model.OutputGenerated("KerasModel.hxx");

   std::cout << "\n\n";
   model.PrintRequiredInputTensors();

   std::cout << "\n\n";
   model.PrintInitializedTensors();

   std::cout << "\n\n";
   model.PrintIntermediateTensors();

   std::cout << "\n\nTensor \"dense/kernel:0\" already exist: " << std::boolalpha
             << model.CheckIfTensorAlreadyExist("dense/kernel:0") << "\n\n";

   std::vector<size_t> tensorShape = model.GetTensorShape("dense/kernel:0");
   std::cout << "Shape of tensor \"dense/kernel:0\": ";
   for (auto &it : tensorShape) {
      std::cout << it << ",";
   }

   std::cout << "\n\nData type of tensor \"dense/kernel:0\": ";
   SOFIE::ETensorType tensorType = model.GetTensorType("dense/kernel:0");
   std::cout << SOFIE::ConvertTypeToString(tensorType);

   std::cout << "\n\n";
   model.PrintGenerated();
}
