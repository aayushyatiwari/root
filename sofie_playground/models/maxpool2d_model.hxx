//Code generated automatically by TMVA for Inference of Model file [maxpool2d_model.pt] at [Wed Mar  4 19:39:00 2026] 

#ifndef ROOT_TMVA_SOFIE_MAXPOOL2D_MODEL
#define ROOT_TMVA_SOFIE_MAXPOOL2D_MODEL

#include <cmath>
#include <vector>
#include "TMVA/SOFIE_common.hxx"

namespace TMVA_SOFIE_maxpool2d_model{
struct Session {

//--- Allocating session memory pool to be used for allocating intermediate tensors
std::vector<char> fIntermediateMemoryPool = std::vector<char>(16);


// --- Positioning intermediate tensor memory --
 // Allocating memory for intermediate tensor 1 with size 16 bytes
float* tensor_1 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

std::vector<float> fVec_op_0_xpad = std::vector<float>(16);

Session(std::string = "") {
}

void doInfer(float const* tensor_input1,  std::vector<float> &output_tensor_1 ){


//----  operator MaxPool  op_0
{
   constexpr int hsize = 4;
   constexpr int hmin = 0;
   constexpr int hmax = 3;
   constexpr int kh = 2;
   constexpr int wsize = 4;
   constexpr int wmin = 0;
   constexpr int wmax = 3;
   constexpr int kw = 2;
   size_t outIndex = 0;
   for (size_t n = 0; n < 1; n++) {
      size_t inputOffset = n*16;
      for (int i = hmin; i < hmax; i+=2) {
         for (int j = wmin; j < wmax; j+=2) {
            float value = -INFINITY;
            for (int l = i;  l < i + kh; l++) {
               if (l < 0 || l >= hsize) continue;
               for (int m = j; m < j + kw; m++) {
                  if (m < 0 || m >= wsize) continue;
                     int index = inputOffset + l*wsize + m;
                     auto xval = tensor_input1[index];
                     if (xval > value) value = xval;
                  }
               }
            tensor_1[outIndex++] = value;
         }
      }
   }
   }
   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;

   FillOutput(tensor_1, output_tensor_1, 4);
}



std::vector<float> infer(float const* tensor_input1){
   std::vector<float > output_tensor_1;
   doInfer(tensor_input1, output_tensor_1 );
   return {output_tensor_1};
}
};   // end of Session

} //TMVA_SOFIE_maxpool2d_model

#endif  // ROOT_TMVA_SOFIE_MAXPOOL2D_MODEL
