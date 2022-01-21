// Paddle-Lite
#include "base_infer.h"
#include "paddle_api.h"
#include "yaml-cpp/yaml.h"

class PaddleLiteInfer : public BaseInfer {
private:
    /* data */
public:
    PaddleLiteInfer(/* args */);
    ~PaddleLiteInfer();

    int load_model(std::string yaml_path);
    int init_inputs_outputs();
    // void init_inputs_outputs(void* input_data);//TODO:
    // void init_inputs_outputs(std::map<std::string, void*>& input_datas);//TODO:
    int init_inputs_outputs(std::string file_path);

    // std::vector<std::string> get_input_names();
    // std::vector<std::string> get_output_names();
    // std::vector<std::string> get_middle_names();

    void init_infer_inputs_outputs();

    void prepare_inputs();  // my_infer->mnn, paddle-lite...
    void infer_model();
    void get_outputs();  // mnn, paddle-lite...->my_infer

    void get_inputs();
    void get_middles();

    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
};
