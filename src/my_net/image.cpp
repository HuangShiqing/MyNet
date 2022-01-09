#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_resize.h"

Image::Image(/* args */) {
}

Image::~Image() {
}

uint8_t* Image::load_image(std::string filename, int dst_w, int dst_h, layout l) {
    int src_w, src_h, src_c;
    uint8_t* data = stbi_load(filename.c_str(), &src_w, &src_h, &src_c, 3);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename.c_str(), stbi_failure_reason());
        exit(0);
    }
    if (src_w != dst_w && src_h != dst_h) {
        uint8_t* data_resized = new uint8_t[dst_w * dst_h * 3];
        stbir_resize(data, src_w, src_h, 0, data_resized, dst_w, dst_h, 0, STBIR_TYPE_UINT8, 3,
                     STBIR_ALPHA_CHANNEL_NONE, 0, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP, STBIR_FILTER_BOX,
                     STBIR_FILTER_BOX, STBIR_COLORSPACE_SRGB, nullptr);
        std::swap(data, data_resized);
        delete data_resized;
    }
    if (l == layout::nchw) {
        uint8_t* data_nchw = new uint8_t[dst_w * dst_h * 3];
        nhwc2nchw(data_nchw, data, dst_w, dst_h);
        std::swap(data, data_nchw);
        delete data_nchw;
    }
    return data;
}

void Image::nhwc2nchw(unsigned char* dst, unsigned char* src, int width, int height) {
    unsigned char* dst_r = dst;
    unsigned char* dst_g = dst + width * height;
    unsigned char* dst_b = dst + 2 * width * height;
    for (int i = 0; i < width * height; i++) {
        dst_r[i] = src[3 * i];
        dst_g[i] = src[3 * i + 1];
        dst_b[i] = src[3 * i + 2];
    }
}