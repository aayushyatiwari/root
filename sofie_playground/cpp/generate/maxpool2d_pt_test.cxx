using namespace TMVA::Experimental;

void maxpool2d_pt_test(){
    std::vector<size_t> inputShape{1, 1, 4, 4};
    std::vector<std::vector<size_t>> inputShapes{inputShape};
    SOFIE::RModel model = SOFIE::PyTorch::Parse("maxpool2d_model.pt", inputShapes);
    model.Generate();
    model.OutputGenerated("maxpool2d_model.hxx");
}

