using namespace TMVA::Experimental;

void elu_pt_test(){
    std::vector<size_t> inputShape{2, 4};
    std::vector<std::vector<size_t>> inputShapes{inputShape};
    SOFIE::RModel model = SOFIE::PyTorch::Parse("elu_model.pt", inputShapes);
    model.Generate();
    model.OutputGenerated("elu_model.hxx");
}
