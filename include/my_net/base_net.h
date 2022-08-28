#pragma once
#include <math.h>
#include <memory.h>

#include "image.h"
#include "base_infer.h"
#include "type.h"

class BaseNet {
private:
    /* data */
public:
    BaseNet(/* args */);
    ~BaseNet();

    void nms(std::vector<Detection>& input, std::vector<Detection>& output, nms_type type, float iou_threshold);

    virtual void init_user_data() = 0;
    virtual void pre_process(uint8_t* data) = 0;
    virtual void forward() = 0;
    virtual void post_process() = 0;
    // virtual std::vector<Detection>& get_detections() = 0;

    BaseInfer* m_infer;
    Image image;
};

// class DetectNet: public BaseNet {
// public:
//     DetectNet();
//     ~DetectNet();

//     virtual std::vector<Detection>& get_detections() = 0;
// };

// class PoseNet: