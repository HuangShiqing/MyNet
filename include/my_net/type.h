#ifndef __TYPE__
#define __TYPE__

typedef enum { 
    coco_17, 
    coco_18,
    body_25
} pose_keypoint_type;

// typedef struct {
//     char keypoints[17][20] = {{"nose"},          {"left_eye"},       {"right_eye"},  {"left_ear"},    {"right_ear"},
//                               {"left_shoulder"}, {"right_shoulder"}, {"left_elbow"}, {"right_elbow"}, {"left_wrist"},
//                               {"right_wrist"},   {"left_hip"},       {"right_hip"},  {"left_knee"},   {"right_knee"},
//                               {"left_ankle"},    {"right_ankle"}};
//     int skeleton[19][2] = {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9},
//                            {8, 10},  {9, 11},  {2, 3},   {1, 2},   {1, 3},   {2, 4},  {3, 5},  {4, 6}, {5, 7}};
// } coco_17_info;
// typedef struct {
//     char keypoints[18][20] = {"Nose",   "Neck",  "R-Sho", "R-Elb",  "R-Wr",  "L-Sho", "L-Elb", "L-Wr",  "R-Hip",
//                               "R-Knee", "R-Ank", "L-Hip", "L-Knee", "L-Ank", "R-Eye", "L-Eye", "R-Ear", "L-Ear"};
//     int skeleton[19][2] = {{1, 2},   {1, 5},   {2, 3}, {3, 4},  {5, 6},   {6, 7},  {1, 8},   {8, 9},  {9, 10}, {1, 11},
//                            {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}, {2, 16}, {5, 17}};
// }coco_18_info;
// typedef struct {
//     char keypoints[26][20] = {"Nose",   "Neck",    "RShoulder", "RElbow", "RWrist",    "LShoulder", "LElbow",
//                               "LWrist", "MidHip",  "RHip",      "RKnee",  "RAnkle",    "LHip",      "LKnee",
//                               "LAnkle", "REye",    "LEye",      "REar",   "LEar",      "LBigToe",   "LSmallToe",
//                               "LHeel",  "RBigToe", "RSmallToe", "RHeel",  "Background"};
//     // TODO:
//     //  int skeleton[19][2]
// } body_25_info;

typedef struct {
    pose_keypoint_type type;
    float x[25];
    float y[25];
    float v[25];
}PoseKeyPoint;

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