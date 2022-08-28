#include <stdio.h>
#include <stdlib.h>
#include "wrapper_openpose.h"

int main(int argc, char const* argv[]) {
    char yaml_path[] = "/home/hsq/DeepLearning/code/MyNet/resource/conf/tensorrt/openpose.yaml";
    char image_path[] = "/home/hsq/DeepLearning/code/MyNet/resource/input/pose.jpg";

    void* p = wrapper_openpose_init(yaml_path);
    uint8_t* data = wrapper_openpose_load_image(p, image_path, 224, 224, nchw);
    wrapper_openpose_pre_process(p, data);
    wrapper_openpose_forward(p);
    wrapper_openpose_post_process(p);
    PoseKeyPoint* d;
    int count;
    wrapper_openpose_get_outputs(p, &d, &count);

    if(count>0) {
        for (int i = 0; i < 18; i++) {
            if (d[0].v[i] == 1) {
                printf("d[%d].x=%f\r\n", i, d[0].x[i]);
                printf("d[%d].y=%f\r\n", i, d[0].y[i]);
            }
        }
    }

    free(data);
    wrapper_openpose_delete(p);
    return 0;
}
