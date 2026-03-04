//Code generated automatically by TMVA for Inference of Model file [elu_model.pt] at [Wed Mar  4 15:15:44 2026] 

#ifndef ROOT_TMVA_SOFIE_ELU_MODEL
#define ROOT_TMVA_SOFIE_ELU_MODEL

#include <algorithm>
#include <cmath>
#include <vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_elu_model{
namespace BLAS{
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
struct Session {
// initialized (weights and constant) tensors
std::vector<float> fTensor_0weight = std::vector<float>(32);
float * tensor_0weight = fTensor_0weight.data();
std::vector<float> fTensor_0bias = std::vector<float>(8);
float * tensor_0bias = fTensor_0bias.data();

//--- Allocating session memory pool to be used for allocating intermediate tensors
std::vector<char> fIntermediateMemoryPool = std::vector<char>(128);


// --- Positioning intermediate tensor memory --
 // Allocating memory for intermediate tensor input0 with size 64 bytes
float* tensor_input0 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor result with size 64 bytes
float* tensor_result = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 64);


Session(std::string filename ="elu_model.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   using TMVA::Experimental::SOFIE::ReadTensorFromStream;
   ReadTensorFromStream(f, tensor_0weight, "tensor_0weight", 32);
   ReadTensorFromStream(f, tensor_0bias, "tensor_0bias", 8);
   f.close();

}

void doInfer(float const* tensor_input1,  std::vector<float> &output_tensor_result ){


//--------- Gemm op_0 { 2 , 4 } * { 8 , 4 } -> { 2 , 8 }
   for (size_t j = 0; j < 2; j++) { 
      size_t y_index = 8 * j;
      for (size_t k = 0; k < 8; k++) { 
         tensor_input0[y_index + k] = tensor_0bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_input0, true, false, 8, 2, 4, 1, tensor_0weight, tensor_input1, 1,nullptr);
   float op_1_alpha = 1;

//------ ELU 
   for (int id = 0; id < 16 ; id++){
      tensor_result[id] = ((tensor_input0[id] >= 0 )? tensor_input0[id] : op_1_alpha * std::exp(tensor_input0[id]) - 1);
   }
   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;

   FillOutput(tensor_result, output_tensor_result, 16);
}



std::vector<float> infer(float const* tensor_input1){
   std::vector<float > output_tensor_result;
   doInfer(tensor_input1, output_tensor_result );
   return {output_tensor_result};
}
};   // end of Session

} //TMVA_SOFIE_elu_model

#endif  // ROOT_TMVA_SOFIE_ELU_MODEL
