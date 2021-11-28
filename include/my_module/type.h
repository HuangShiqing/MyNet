#ifndef __TYPE__
#define __TYPE__

typedef struct {
    int x1;
    int y1;
    int x2;
    int y2;
    int x_center;
    int y_center;
    int w;
    int h;
} BBox;

typedef struct {
    BBox box;
    float confidence;
    char label[10];
} Detection;

typedef enum {
    hard_nms = 1,
    blending_nms = 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/
} nms_type;

typedef enum { 
    nchw, 
    nhwc 
} layout;

#endif