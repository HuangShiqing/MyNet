#include "rfb320.h"
#include "mnn_infer.h"

RFB320::RFB320(std::string yaml_path) {
    // 1.init model
    m_infer = new MNNInfer();
    m_infer->load_model(yaml_path);
    // 2.init inputs
    m_infer->init_infer_inputs_outputs();
    // 3.init user data
    init_user_data();
}

RFB320::~RFB320() {
    delete m_infer;
    // ((MNNInfer*)m_infer)->~MNNInfer();
}

void RFB320::init_user_data() {
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))
    int in_w = 320;
    int in_h = 240;
    int num_featuremap = 4;
    const std::vector<std::vector<float>> min_boxes = {
        {10.0f, 16.0f, 24.0f}, {32.0f, 48.0f}, {64.0f, 96.0f}, {128.0f, 192.0f, 256.0f}};
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<int> w_h_list = {in_w, in_h};
    for (auto size : w_h_list) {
        std::vector<float> fm_item;
        for (float stride : strides) {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }
    for (auto size : w_h_list) {
        shrinkage_size.push_back(strides);
    }
    /* generate prior anchors */
    for (int index = 0; index < num_featuremap; index++) {
        float scale_w = in_w / shrinkage_size[0][index];
        float scale_h = in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;
                for (float k : min_boxes[index]) {
                    float w = k / in_w;
                    float h = k / in_h;
                    m_user_data.priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                }
            }
        }
    }
    /* generate prior anchors finished */
    m_user_data.num_anchors = m_user_data.priors.size();
}

void RFB320::pre_process(uint8_t* data) {
    float mean = 127;
    float norm = 1.0 / 128;
    auto input_tensor = m_infer->input_tensors_["input"];
    auto shape = input_tensor.shape_;
    int count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

    for (int i = 0; i < count; i++) {
        *((float*)(input_tensor.data_)+i) = (*(data + i) - mean) * norm;
    }
}

void RFB320::forward() {
    m_infer->prepare_inputs();
    m_infer->infer_model();
    m_infer->get_outputs();
}

void RFB320::post_process() {
    float* boxes_data = (float*)m_infer->output_tensors_["boxes"].data_;
    float* scores_data = (float*)m_infer->output_tensors_["scores"].data_;
    std::vector<Detection> detections;
    for (int i = 0; i < m_user_data.num_anchors; i++) {
        if (scores_data[i * 2 + 1] > m_user_data.score_threshold) {
            Detection det;
            float x_center =
                boxes_data[i * 4] * m_user_data.center_variance * m_user_data.priors[i][2] + m_user_data.priors[i][0];
            float y_center = boxes_data[i * 4 + 1] * m_user_data.center_variance * m_user_data.priors[i][3] +
                             m_user_data.priors[i][1];
            float w = exp(boxes_data[i * 4 + 2] * m_user_data.size_variance) * m_user_data.priors[i][2];
            float h = exp(boxes_data[i * 4 + 3] * m_user_data.size_variance) * m_user_data.priors[i][3];

            det.box.x1 = clip(x_center - w / 2.0, 1) * m_user_data.image_w;
            det.box.y1 = clip(y_center - h / 2.0, 1) * m_user_data.image_h;
            det.box.x2 = clip(x_center + w / 2.0, 1) * m_user_data.image_w;
            det.box.y2 = clip(y_center + h / 2.0, 1) * m_user_data.image_h;
            det.confidence = clip(scores_data[i * 2 + 1], 1);
            strcpy(det.label, "face");
            detections.push_back(det);
        }
    }
    m_detections.clear();
    nms(detections, m_detections, hard_nms, 0.3);
}

std::vector<Detection>& RFB320::get_detections() {
    return this->m_detections;
}