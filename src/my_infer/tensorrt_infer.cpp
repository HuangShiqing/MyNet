#include "tensorrt_infer.h"
// https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#c_topics

TensorRTInfer::TensorRTInfer(/* args */) {
}

TensorRTInfer::~TensorRTInfer() {
    delete network_;
    delete buffers_;
}

int TensorRTInfer::load_model(std::string yaml_path) {
    YAML::Node yaml_node = YAML::LoadFile(yaml_path);
    std::string model_path = yaml_node["model_path"].as<std::string>();

    auto builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network_ = builder->createNetworkV2(explicitBatch);
    auto config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(16_MiB);
    config->setFlag(BuilderFlag::kFP16);
    // config->setFlag(BuilderFlag::kINT8);
    // samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    // samplesCommon::enableDLA(builder, config, false);//DLA

    // TODO: caffe parser
    auto parser = nvonnxparser::createParser(*network_, sample::gLogger.getTRTLogger());
    auto parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    config->setProfileStream(*profileStream);
    auto plan = builder->buildSerializedNetwork(*network_, *config);
    auto runtime = createInferRuntime(sample::gLogger.getTRTLogger());
    // auto engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                         samplesCommon::InferDeleter());
    context_ = engine->createExecutionContext();
    buffers_ = new samplesCommon::BufferManager(engine);

    delete parser;
    delete config;
    delete builder;
}

int TensorRTInfer::init_inputs_outputs() {
    // TODO:
}

int TensorRTInfer::init_inputs_outputs(std::string file_path) {
    // TODO:
}

void TensorRTInfer::init_infer_inputs_outputs() {
    std::vector<std::string> input_names;
    for (int index = 0; index < network_->getNbInputs(); index++) {
        auto tensor = network_->getInput(index);

        auto name = tensor->getName();
        auto dims = tensor->getDimensions();

        std::vector<int> tmp_shape;
        for (int i = 0; i < dims.nbDims; i++) {
            tmp_shape.push_back(dims.d[i]);
        }
        int count = std::accumulate(tmp_shape.begin(), tmp_shape.end(), 1, std::multiplies<int>());

        Tensor t;
        t.name_ = name;
        // t.dtype_ =
        t.shape_ = tmp_shape;
        t.data_size_ = count * sizeof(float);
        t.data_ = static_cast<float*>(buffers_->getHostBuffer(name));  // link
        // t.lod_ =
        input_tensors_.insert(std::make_pair(name, t));
    }

    for (int index = 0; index < network_->getNbOutputs(); index++) {
        auto tensor = network_->getOutput(index);

        auto name = tensor->getName();
        auto dims = tensor->getDimensions();

        std::vector<int> tmp_shape;
        for (int i = 0; i < dims.nbDims; i++) {
            tmp_shape.push_back(dims.d[i]);
        }
        int count = std::accumulate(tmp_shape.begin(), tmp_shape.end(), 1, std::multiplies<int>());

        Tensor t;
        t.name_ = name;
        // t.dtype_ =
        t.shape_ = tmp_shape;
        t.data_size_ = count * sizeof(float);
        t.data_ = static_cast<float*>(buffers_->getHostBuffer(name));
        ;  // link
        // t.lod_ =
        output_tensors_.insert(std::make_pair(name, t));
    }
}

void TensorRTInfer::prepare_inputs() {
    // this function only for copy mode or link mode's user copy work
    // Memcpy from host input buffers to device input buffers
    buffers_->copyInputToDevice();
}

void TensorRTInfer::infer_model() {
    bool status = context_->executeV2(buffers_->getDeviceBindings().data());
};

void TensorRTInfer::get_outputs() {
    // this function only for copy mode or link mode's user copy work
    buffers_->copyOutputToHost();
}

void TensorRTInfer::get_inputs(){
    // TODO:
};

void TensorRTInfer::get_middles() {
    // TODO:
}