#pragma once
#include <math.h>
#include <memory.h>
#include <algorithm>

#include "image.h"
#include "my_net.h"

struct BBox {
    int x1;
    int y1;
    int x2;
    int y2;
    int x_center;
    int y_center;
    int w;
    int h;
};
struct Detection {
    BBox box;
    float confidence;
    std::string label;
};
enum class nms_type {
    hard_nms = 1,
    blending_nms = 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/
};

class BaseModule {
private:
    /* data */
public:
    BaseModule(/* args */);
    ~BaseModule();

    void nms(std::vector<Detection>& input, std::vector<Detection>& output, nms_type type, float iou_threshold);

    void init_user_data();
    void pre_process(uint8_t* data);
    void forward();
    void post_process();

    MyNet* m_net;
    Image image;
};