#pragma once
#include "base_net.h"

class OpenPose : public BaseNet {
private:
    /* data */
public:
    OpenPose(std::string yaml_path);
    ~OpenPose();

    void pre_process(uint8_t* data);
    void forward();
    void post_process();
    std::vector<PoseKeyPoint>& get_outputs();

private:
    void init_user_data();
    struct user_data {
        int image_w = 224;
        int image_h = 224;

        // data
        float cmap_threshold = 0.1;
        float link_threshold = 0.1;
        int cmap_window = 5;
        int line_integral_samples = 7;
        int max_num_objects = 100;
        int M = 100;

        int N;
        int C;
        int H;
        int W;
	    int topology[84] = { 0, 1, 15, 13, 2, 3, 13, 11, 4, 5, 16, 14, 6, 7, 14, 12, 8, 9, 11, 12, 10, 11, 5, 7, 12, 13, 6, 8, 14, 15, 7, 9, 16, 17, 8, 10, 18, 19, 1, 2, 20, 21, 0, 1, 22, 23, 0, 2, 24, 25, 1, 3, 26, 27, 2, 4, 28, 29, 3, 5, 30, 31, 4, 6, 32, 33, 17, 0, 34, 35, 17, 5, 36, 37, 17, 6, 38, 39, 17, 11, 40, 41, 17, 12};
    };
    user_data m_user_data;
    std::vector<PoseKeyPoint> m_pose_keypoints;
};


void find_peaks_out_nchw(int* counts,         // C
                         int* peaks,          // CxMx2
                         const float* input,  // CxHxW
                         const int N, const int C, const int H, const int W, const int M, const float threshold,
                         const int window_size);
void refine_peaks_out_nchw(float* refined_peaks,  // NxCxMx2
                           const int* counts,     // NxC
                           const int* peaks,      // NxCxMx2
                           const float* cmap, const int N, const int C, const int H, const int W, const int M,
                           const int window_size);
void paf_score_graph_out_nkhw(float* score_graph,   // NxKxMxM
                              const int* topology,  // Kx4
                              const float* paf,     // Nx2KxHxW
                              const int* counts,    // NxC
                              const float* peaks,   // NxCxMx2
                              const int N, const int K, const int C, const int H, const int W, const int M,
                              const int num_integral_samples);
std::size_t assignment_out_workspace(const int M);
void assignment_out_nk(int* connections,          // NxKx2xM
                       const float* score_graph,  // NxKxMxM
                       const int* topology,       // Kx4
                       const int* counts,         // NxC
                       const int N, const int C, const int K, const int M, const float score_threshold,
                       void* workspace);
std::size_t connect_parts_out_workspace(int C, int M);
void connect_parts_out_batch(int* object_counts,      // N
                             int* objects,            // NxPxC
                             const int* connections,  // NxKx2xM
                             const int* topology,     // Kx4
                             const int* counts,       // NxC
                             const int N, const int K, const int C, const int M, const int P, void* workspace);