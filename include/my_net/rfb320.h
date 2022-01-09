#pragma once
#include "base_net.h"

class RFB320 : public BaseNet {
private:
    /* data */
public:
    RFB320(std::string model_file);
    ~RFB320();

    void pre_process(uint8_t* data);
    void forward();
    void post_process();
    std::vector<Detection>& get_detections();

private:
    void init_user_data();
    struct user_data {
        /* data */
        float score_threshold = 0.65;
        float center_variance = 0.1;
        float size_variance = 0.2;
        int image_w = 320;
        int image_h = 240;

        std::vector<std::vector<float>> priors = {};
        int num_anchors = 0;
    };
    user_data m_user_data;
    std::vector<Detection> m_detections;
};
