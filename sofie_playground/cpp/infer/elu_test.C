#include "/home/imaayush/code/root_src/sofie_playground/models/elu_model.hxx"
void elu_test() {
    float input[] = { 1.0, -1.0, 0.5, -0.5,
                      2.0, -2.0, 1.5, -1.5 };
    float expected[] = { 0.06505805, -0.32683983,  0.8157425,   0.76650894,
                        0.20902929, -0.03452254, -0.02935613,  0.82148886,
                        -0.41964692, -0.6812308,   1.9643464,   1.3798876,
                        0.15969864, -0.5236627,  -0.03393746,  2.2388086 };
    TMVA_SOFIE_elu_model::Session s;
    auto output = s.infer(input);

    float tol = 1e-5;
    bool pass = true;
    for (int i = 0; i < 16; i++) {
        float diff = std::abs(output[i] - expected[i]);
        if (diff > tol) {
            std::cout << "FAIL at [" << i << "]: got " << output[i]
                      << " expected " << expected[i] << " diff " << diff << "\n";
            pass = false;
        }
    }
    if (pass) std::cout << "ELU: ALL PASSED\n";
}
