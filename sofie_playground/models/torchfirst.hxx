//Code generated automatically by TMVA for Inference of Model file [torchfirst.onnx] at [Tue Mar  3 13:54:31 2026] 

#ifndef ROOT_TMVA_SOFIE_TORCHFIRST
#define ROOT_TMVA_SOFIE_TORCHFIRST

#include <algorithm>
#include <vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_torchfirst{
namespace BLAS{
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
struct Session {
// initialized (weights and constant) tensors
std::vector<float> fTensor_2bias = std::vector<float>(1);
float * tensor_2bias = fTensor_2bias.data();
std::vector<float> fTensor_2weight = std::vector<float>(4);
float * tensor_2weight = fTensor_2weight.data();
std::vector<float> fTensor_0bias = std::vector<float>(4);
float * tensor_0bias = fTensor_0bias.data();
std::vector<float> fTensor_0weight = std::vector<float>(12);
float * tensor_0weight = fTensor_0weight.data();

//--- Allocating session memory pool to be used for allocating intermediate tensors
std::vector<char> fIntermediateMemoryPool = std::vector<char>(20);


// --- Positioning intermediate tensor memory --
 // Allocating memory for intermediate tensor relu with size 16 bytes
float* tensor_relu = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor linear_1 with size 4 bytes
float* tensor_linear_1 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 16);


Session(std::string filename ="torchfirst.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   using TMVA::Experimental::SOFIE::ReadTensorFromStream;
   ReadTensorFromStream(f, tensor_2bias, "tensor_2bias", 1);
   ReadTensorFromStream(f, tensor_2weight, "tensor_2weight", 4);
   ReadTensorFromStream(f, tensor_0bias, "tensor_0bias", 4);
   ReadTensorFromStream(f, tensor_0weight, "tensor_0weight", 12);
   f.close();

}

void doInfer(float const* tensor_input,  std::vector<float> &output_tensor_linear_1 ){


//--------- Gemm op_0 { 1 , 3 } * { 4 , 3 } -> { 1 , 4 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = 4 * j;
      for (size_t k = 0; k < 4; k++) { 
         tensor_relu[y_index + k] = tensor_0bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_relu, true, false, 4, 1, 3, 1, tensor_0weight, tensor_input, 1,nullptr);
   for (int id = 0; id < 4 ; id++){
      tensor_relu[id] = ((tensor_relu[id] > 0 )? tensor_relu[id] : 0);
   }

//--------- Gemm op_1 { 1 , 4 } * { 1 , 4 } -> { 1 , 1 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = j;
      for (size_t k = 0; k < 1; k++) { 
         tensor_linear_1[y_index + k] = tensor_2bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_linear_1, true, false, 1, 1, 4, 1, tensor_2weight, tensor_relu, 1,nullptr);
   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;

   FillOutput(tensor_linear_1, output_tensor_linear_1, 1);
}



std::vector<float> infer(float const* tensor_input){
   std::vector<float > output_tensor_linear_1;
   doInfer(tensor_input, output_tensor_linear_1 );
   return {output_tensor_linear_1};
}
};   // end of Session

} //TMVA_SOFIE_torchfirst

#endif  // ROOT_TMVA_SOFIE_TORCHFIRST
