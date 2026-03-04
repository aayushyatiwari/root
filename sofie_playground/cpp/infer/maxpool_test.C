#include "/home/imaayush/code/root_src/sofie_playground/models/maxpool2d_model.hxx"
void maxpool_test() {
    float input[] = { 1.0,  3.0,  2.0,  4.0,
                      5.0,  7.0,  6.0,  8.0,
                      9.0, 11.0, 10.0, 12.0,
                     13.0, 15.0, 14.0, 16.0 };

    float expected[] = { 7.0, 8.0, 15.0, 16.0 };

    TMVA_SOFIE_maxpool2d_model::Session s;
    auto output = s.infer(input);

    float tol = 1e-5;
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        float diff = std::abs(output[i] - expected[i]);
        if (diff > tol) {
            std::cout << "FAIL at [" << i << "]: got " << output[i]
                      << " expected " << expected[i] << " diff " << diff << "\n";
            pass = false;
        }
    }
    if (pass) std::cout << "MaxPool2D: ALL PASSED\n";
}
