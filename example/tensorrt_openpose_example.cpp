#include <iostream>
#include <vector>

#include "openpose.h"

int main(int argc, char const* argv[]) {
    std::string yaml_path = "/home/hsq/DeepLearning/code/MyNet/resource/conf/tensorrt/openpose.yaml";
    std::string image_path = "/home/hsq/DeepLearning/code/MyNet/resource/input/pose.jpg";

    OpenPose openpose(yaml_path);
    uint8_t* data = openpose.image.load_image(image_path, 224, 224);
    openpose.pre_process(data);
    openpose.forward();
    openpose.post_process();
    std::vector<PoseKeyPoint> d = openpose.get_outputs();
    if (d.size()) {
        for (int i = 0; i < 18; i++) {
            if (d[0].v[i] == 1) {
                std::cout << "d[" << i << "].x=" << d[0].x[i] << std::endl;
                std::cout << "d[" << i << "].y=" << d[0].y[i] << std::endl;
            }
        }
    }

    delete[] data;
    return 0;
}