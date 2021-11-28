#include "base_module.h"

#include <algorithm>

BaseModule::BaseModule(/* args */) {
}

BaseModule::~BaseModule() {
}

void BaseModule::nms(std::vector<Detection>& input, std::vector<Detection>& output, nms_type type,
                     float iou_threshold) {
    std::sort(input.begin(), input.end(),
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    int box_num = input.size();
    std::vector<int> merged(box_num, 0);
    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<Detection> buf;
        buf.push_back(input[i]);
        merged[i] = 1;

        float h0 = input[i].box.y2 - input[i].box.y1 + 1;
        float w0 = input[i].box.x2 - input[i].box.x1 + 1;
        float area0 = h0 * w0;
        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;
            float inner_x0 = input[i].box.x1 > input[j].box.x1 ? input[i].box.x1 : input[j].box.x1;
            float inner_y0 = input[i].box.y1 > input[j].box.y1 ? input[i].box.y1 : input[j].box.y1;
            float inner_x1 = input[i].box.x2 < input[j].box.x2 ? input[i].box.x2 : input[j].box.x2;
            float inner_y1 = input[i].box.y2 < input[j].box.y2 ? input[i].box.y2 : input[j].box.y2;
            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;
            if (inner_h <= 0 || inner_w <= 0)
                continue;
            float inner_area = inner_h * inner_w;
            float h1 = input[j].box.y2 - input[j].box.y1 + 1;
            float w1 = input[j].box.x2 - input[j].box.x1 + 1;
            float area1 = h1 * w1;
            float score;
            score = inner_area / (area0 + area1 - inner_area);
            if (score > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        switch (type) {
            case nms_type::hard_nms: {
                output.push_back(buf[0]);
                break;
            }
            case nms_type::blending_nms: {
                float total = 0;
                for (unsigned int i = 0; i < buf.size(); i++) {
                    total += exp(buf[i].confidence);
                }
                Detection rects;
                memset(&rects, 0, sizeof(rects));
                for (unsigned int i = 0; i < buf.size(); i++) {
                    float rate = exp(buf[i].confidence) / total;
                    rects.box.x1 += buf[i].box.x1 * rate;
                    rects.box.y1 += buf[i].box.y1 * rate;
                    rects.box.x2 += buf[i].box.x2 * rate;
                    rects.box.y2 += buf[i].box.y2 * rate;
                    rects.confidence += buf[i].confidence * rate;
                }
                strcpy(rects.label, buf[0].label);
                output.push_back(rects);
                break;
            }
            default: {
                printf("wrong type of nms.");
                exit(-1);
            }
        }
    }
}