#pragma once
#include <string>

enum class layout { 
    nchw, 
    nhwc 
};

class Image {
private:
    /* data */
public:
    Image(/* args */);
    ~Image();

    uint8_t* load_image(std::string filename, int w, int h, layout layout = layout::nchw);
    void nhwc2nchw(unsigned char* dst, unsigned char* src, int width, int height);
};
