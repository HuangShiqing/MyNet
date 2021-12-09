#include <stdio.h>

#include "wrapper_rfb320.h"

int main(int argc, char const* argv[]) {
    char model_path[] = "/home/hsq/DeepLearning/code/MyNet/resource/model/RFB-320.mnn";
    char image_path[] = "/home/hsq/DeepLearning/code/MyNet/resource/face.jpg";

    void* p = wrapper_rfb320_init(model_path);
    uint8_t* data = wrapper_rfb320_load_image(p, image_path, 320, 240, nchw);
    wrapper_rfb320_pre_process(p, data);
    wrapper_rfb320_forward(p);
    wrapper_rfb320_post_process(p);
    Detection* d;
    int count;
    wrapper_rfb320_get_detections(p, &d, &count);

    printf("d[0].box.x1=%d\r\n", d[0].box.x1);
    printf("d[0].box.y1=%d\r\n", d[0].box.y1);
    printf("d[0].box.x2=%d\r\n", d[0].box.x2);
    printf("d[0].box.y2=%d\r\n", d[0].box.y2);

    free(data);
    wrapper_rfb320_delete(p);
    return 0;
}
