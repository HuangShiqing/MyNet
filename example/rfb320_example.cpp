#include <iostream>
#include <vector>

#include "rfb320.h"

int main(int argc, char const* argv[]) {
    std::string yaml_path = "/home/hsq/DeepLearning/code/MyNet/resource/conf/mnn/rfb320.yaml";
    std::string image_path = "/home/hsq/DeepLearning/code/MyNet/resource/face.jpg";

    RFB320 rfb320(yaml_path);
    uint8_t* data = rfb320.image.load_image(image_path, 320, 240);
    rfb320.pre_process(data);
    rfb320.forward();
    rfb320.post_process();
    std::vector<Detection> d = rfb320.get_detections();

    if (d.size()) {
        std::cout << "d[0].box.x1=" << d[0].box.x1 << std::endl;
        std::cout << "d[0].box.y1=" << d[0].box.y1 << std::endl;
        std::cout << "d[0].box.x2=" << d[0].box.x2 << std::endl;
        std::cout << "d[0].box.y2=" << d[0].box.y2 << std::endl;
    }

    delete[] data;
    return 0;
}
