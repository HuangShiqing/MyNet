#include "openpose.h"

#include <queue>

#include "tensorrt_infer.h"

OpenPose::OpenPose(std::string yaml_path) {
    // 1.init model
    m_infer = new TensorRTInfer();
    m_infer->load_model(yaml_path);
    // 2.init inputs
    m_infer->init_infer_inputs_outputs();
    // 3.init user data
    init_user_data();
}

OpenPose::~OpenPose() {
    delete m_infer;
}

void OpenPose::init_user_data() {
    m_user_data.N = m_infer->output_tensors_["output_0"].shape_[0];
    m_user_data.C = m_infer->output_tensors_["output_0"].shape_[1];
    m_user_data.H = m_infer->output_tensors_["output_0"].shape_[2];
    m_user_data.W = m_infer->output_tensors_["output_0"].shape_[3];
}

void OpenPose::pre_process(uint8_t* data) {
    // mean = {0.485, 0.456, 0.406}
    // std = {0.229, 0.224, 0.225}
    // (data/255-mean)/std 
    float mean[3] = {123.6, 116.3, 103.5};//0
    float norm[3] = {0.0171, 0.0175, 0.0174};//1.0 / 255;
    auto input_tensor = m_infer->input_tensors_["input_0"];
    auto shape = input_tensor.shape_;
    int count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

    for (int i = 0; i < count; i+=3) {
        *((float*)(input_tensor.data_) + i) = (*(data + i) - mean[0]) * norm[0];
        *((float*)(input_tensor.data_) + i+1) = (*(data + i+1) - mean[1]) * norm[1];
        *((float*)(input_tensor.data_) + i+2) = (*(data + i+2) - mean[2]) * norm[2];
    }
    // for (int i = 0; i < count; i++) {
    //     *((float*)(input_tensor.data_) + i) = (*(data + i) - mean) * norm;
    // }
}

void OpenPose::forward() {
    m_infer->prepare_inputs();
    m_infer->infer_model();
    m_infer->get_outputs();
}

void OpenPose::post_process() {
    float* cmap_data = (float*)m_infer->output_tensors_["output_0"].data_;
    float* paf_data = (float*)m_infer->output_tensors_["output_1"].data_;

    // 1. Find peaks (NMS)
    std::vector<int> peaks, peak_counts;
    size_t peak_size = m_user_data.N * m_user_data.C * m_user_data.M * 2;  // NxCxMx2
    peaks.resize(peak_size);
    size_t peak_count_size = m_user_data.N * m_user_data.C;  // NxC
    peak_counts.resize(peak_count_size);
    int* peaks_ptr = &peaks[0];              // return value
    int* peak_counts_ptr = &peak_counts[0];  // return value

    find_peaks_out_nchw(peak_counts_ptr, peaks_ptr, cmap_data, m_user_data.N, m_user_data.C, m_user_data.H,
                        m_user_data.W, m_user_data.M, m_user_data.cmap_threshold, m_user_data.cmap_window);

    // 2. Refine peaks
    std::vector<float> refined_peaks;  // NxCxMx2
    refined_peaks.resize(peak_size);
    for (int i = 0; i < refined_peaks.size(); i++) {
        refined_peaks[0] = 0;
    }
    float* refined_peaks_ptr = &refined_peaks[0];  // return value

    refine_peaks_out_nchw(refined_peaks_ptr, peak_counts_ptr, peaks_ptr, cmap_data, m_user_data.N, m_user_data.C,
                          m_user_data.H, m_user_data.W, m_user_data.M, m_user_data.cmap_window);

    // 3. Score paf
    int K = 21;
    size_t score_graph_size = m_user_data.N * K * m_user_data.M * m_user_data.M;  // NxKxMxM
    std::vector<float> score_graph;
    score_graph.resize(score_graph_size);
    float* score_graph_ptr = &score_graph[0];

    paf_score_graph_out_nkhw(score_graph_ptr, m_user_data.topology, paf_data, peak_counts_ptr, refined_peaks_ptr,
                             m_user_data.N, K, m_user_data.C, m_user_data.H, m_user_data.W, m_user_data.M,
                             m_user_data.line_integral_samples);

    // 4. Assignment algorithm
    std::vector<int> connections;
    connections.resize(m_user_data.N * K * 2 * m_user_data.M);
    for (int i = 0; i < connections.size(); i++) {
        connections[i] = -1;
    }
    int* connections_ptr = &connections[0];

    void* workspace = (void*)malloc(assignment_out_workspace(m_user_data.M));
    assignment_out_nk(connections_ptr, score_graph_ptr, m_user_data.topology, peak_counts_ptr, m_user_data.N,
                      m_user_data.C, K, m_user_data.M, m_user_data.link_threshold, workspace);
    free(workspace);

    // 5. Merging
    std::vector<int> objects;
    objects.resize(m_user_data.N * m_user_data.max_num_objects * m_user_data.C);
    for (int i = 0; i < objects.size(); i++) {
        objects[i] = -1;
    }
    int* objects_ptr = &objects[0];

    std::vector<int> object_counts;
    object_counts.resize(m_user_data.N);
    object_counts[0] = 0;  // batchSize=1
    int* object_counts_ptr = &object_counts[0];

    void* merge_workspace = malloc(connect_parts_out_workspace(m_user_data.C, m_user_data.M));
    connect_parts_out_batch(object_counts_ptr, objects_ptr, connections_ptr, m_user_data.topology, peak_counts_ptr,
                            m_user_data.N, K, m_user_data.C, m_user_data.M, m_user_data.max_num_objects,
                            merge_workspace);
    free(merge_workspace);

    m_pose_keypoints.clear();
    for (int i = 0; i < object_counts[0]; i++) {
        int* obj = &objects_ptr[m_user_data.C * i];
        PoseKeyPoint pose;
        pose.type = pose_keypoint_type::coco_18;
        for (int j = 0; j < m_user_data.C; j++) {
            int k = (int)obj[j];
            if (k >= 0) {
                float* peak = &refined_peaks_ptr[j * m_user_data.M * 2];
                int x = (int)(peak[k * 2 + 1] * m_user_data.image_w);
                int y = (int)(peak[k * 2] * m_user_data.image_h);
                // std::cout << x << "," << y << std::endl;
                pose.x[j] = x;
                pose.y[j] = y;
                pose.v[j] = 1;
            }
            else {
                pose.x[j] = 0;
                pose.y[j] = 0;
                pose.v[j] = 0;
            }
        }
        m_pose_keypoints.push_back(pose);
    }
}

std::vector<PoseKeyPoint>& OpenPose::get_outputs() {
    return this->m_pose_keypoints;
}

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
void find_peaks_out_hw(int* counts,         // 1
                       int* peaks,          // Mx2
                       const float* input,  // HxW
                       const int H, const int W, const int M, const float threshold, const int window_size) {
    int win = window_size / 2;
    int count = 0;

    for (int i = 0; i < H && count < M; i++) {
        for (int j = 0; j < W && count < M; j++) {
            float val = input[i * W + j];
            // skip if below threshold
            if (val < threshold)
                continue;
            // compute window bounds
            int ii_min = MAX(i - win, 0);
            int jj_min = MAX(j - win, 0);
            int ii_max = MIN(i + win + 1, H);
            int jj_max = MIN(j + win + 1, W);
            // search for larger value in window
            bool is_peak = true;
            for (int ii = ii_min; ii < ii_max; ii++) {
                for (int jj = jj_min; jj < jj_max; jj++) {
                    if (input[ii * W + jj] > val) {
                        is_peak = false;
                    }
                }
            }
            // add peak
            if (is_peak) {
                peaks[count * 2] = i;
                peaks[count * 2 + 1] = j;
                count++;
            }
        }
    }

    *counts = count;
}

void find_peaks_out_chw(int* counts,         // C
                        int* peaks,          // CxMx2
                        const float* input,  // CxHxW
                        const int C, const int H, const int W, const int M, const float threshold,
                        const int window_size) {
    for (int c = 0; c < C; c++) {
        int* counts_c = &counts[c];
        int* peaks_c = &peaks[c * M * 2];
        const float* input_c = &input[c * H * W];
        find_peaks_out_hw(counts_c, peaks_c, input_c, H, W, M, threshold, window_size);
    }
}

void find_peaks_out_nchw(int* counts,         // C
                         int* peaks,          // CxMx2
                         const float* input,  // CxHxW
                         const int N, const int C, const int H, const int W, const int M, const float threshold,
                         const int window_size) {
    for (int n = 0; n < N; n++) {
        int* counts_n = &counts[n * C];
        int* peaks_n = &peaks[n * C * M * 2];
        const float* input_n = &input[n * C * H * W];
        find_peaks_out_chw(counts_n, peaks_n, input_n, C, H, W, M, threshold, window_size);
    }
}

inline int reflect(int idx, int min, int max) {
    if (idx < min) {
        return -idx;
    } else if (idx >= max) {
        return max - (idx - max) - 2;
    } else {
        return idx;
    }
}

void refine_peaks_out_hw(float* refined_peaks,  // Mx2
                         const int* counts,     // 1
                         const int* peaks,      // Mx2
                         const float* cmap,     // HxW
                         const int H, const int W, const int M, const int window_size) {
    int count = *counts;
    int win = window_size / 2;

    for (int m = 0; m < count; m++) {
        float* refined_peak = &refined_peaks[m * 2];
        refined_peak[0] = 0.;
        refined_peak[1] = 0.;
        const int* peak = &peaks[m * 2];

        int i = peak[0];
        int j = peak[1];
        float weight_sum = 0.;

        for (int ii = i - win; ii < i + win + 1; ii++) {
            int ii_idx = reflect(ii, 0, H);
            for (int jj = j - win; jj < j + win + 1; jj++) {
                int jj_idx = reflect(jj, 0, W);

                float weight = cmap[ii_idx * W + jj_idx];
                refined_peak[0] += weight * ii;
                refined_peak[1] += weight * jj;
                weight_sum += weight;
            }
        }

        refined_peak[0] /= weight_sum;
        refined_peak[1] /= weight_sum;
        refined_peak[0] += 0.5;  // center pixel
        refined_peak[1] += 0.5;  // center pixel
        refined_peak[0] /= H;    // normalize coordinates
        refined_peak[1] /= W;    // normalize coordinates
    }
}

void refine_peaks_out_chw(float* refined_peaks,  // CxMx2
                          const int* counts,     // C
                          const int* peaks,      // CxMx2
                          const float* cmap, const int C, const int H, const int W, const int M,
                          const int window_size) {
    for (int c = 0; c < C; c++) {
        refine_peaks_out_hw(&refined_peaks[c * M * 2], &counts[c], &peaks[c * M * 2], &cmap[c * H * W], H, W, M,
                            window_size);
    }
}

void refine_peaks_out_nchw(float* refined_peaks,  // NxCxMx2
                           const int* counts,     // NxC
                           const int* peaks,      // NxCxMx2
                           const float* cmap, const int N, const int C, const int H, const int W, const int M,
                           const int window_size) {
    for (int n = 0; n < N; n++) {
        refine_peaks_out_chw(&refined_peaks[n * C * M * 2], &counts[n * C], &peaks[n * C * M * 2], &cmap[n * C * H * W],
                             C, H, W, M, window_size);
    }
}

#define EPS 1e-5
void paf_score_graph_out_hw(float* score_graph,  // MxM
                            const float* paf_i,  // HxW
                            const float* paf_j,  // HxW
                            const int counts_a, const int counts_b,
                            const float* peaks_a,  // Mx2
                            const float* peaks_b,  // Mx2
                            const int H, const int W, const int M, const int num_integral_samples) {
    for (int a = 0; a < counts_a; a++) {
        // compute point A
        float pa_i = peaks_a[a * 2] * H;
        float pa_j = peaks_a[a * 2 + 1] * W;

        for (int b = 0; b < counts_b; b++) {
            // compute point B
            float pb_i = peaks_b[b * 2] * H;
            float pb_j = peaks_b[b * 2 + 1] * W;

            // compute vector A->B
            float pab_i = pb_i - pa_i;
            float pab_j = pb_j - pa_j;

            // compute normalized vector A->B
            float pab_norm = sqrtf(pab_i * pab_i + pab_j * pab_j) + EPS;
            float uab_i = pab_i / pab_norm;
            float uab_j = pab_j / pab_norm;

            float integral = 0.;
            float increment = 1.f / num_integral_samples;

            for (int t = 0; t < num_integral_samples; t++) {
                // compute integral point T
                float progress = (float)t / ((float)num_integral_samples - 1);
                float pt_i = pa_i + progress * pab_i;
                float pt_j = pa_j + progress * pab_j;

                // convert to int
                // note: we do not need to subtract 0.5 when indexing, because
                // round(x - 0.5) = int(x)
                int pt_i_int = (int)pt_i;
                int pt_j_int = (int)pt_j;

                // skip point if out of bounds (will weaken integral)
                if (pt_i_int < 0)
                    continue;
                if (pt_i_int >= H)
                    continue;
                if (pt_j_int < 0)
                    continue;
                if (pt_j_int >= W)
                    continue;

                // get vector at integral point from PAF
                float pt_paf_i = paf_i[pt_i_int * W + pt_j_int];
                float pt_paf_j = paf_j[pt_i_int * W + pt_j_int];

                // compute dot product of normalized A->B with PAF vector at integral
                // point
                float dot = pt_paf_i * uab_i + pt_paf_j * uab_j;
                integral += dot;
            }

            integral /= num_integral_samples;
            score_graph[a * M + b] = integral;
        }
    }
}

void paf_score_graph_out_khw(float* score_graph,   // KxMxM
                             const int* topology,  // Kx4
                             const float* paf,     // 2KxHxW
                             const int* counts,    // C
                             const float* peaks,   // CxMx2
                             const int K, const int C, const int H, const int W, const int M,
                             const int num_integral_samples) {
    for (int k = 0; k < K; k++) {
        float* score_graph_k = &score_graph[k * M * M];
        const int* tk = &topology[k * 4];
        const int paf_i_idx = tk[0];
        const int paf_j_idx = tk[1];
        const int cmap_a_idx = tk[2];
        const int cmap_b_idx = tk[3];
        const float* paf_i = &paf[paf_i_idx * H * W];
        const float* paf_j = &paf[paf_j_idx * H * W];

        const int counts_a = counts[cmap_a_idx];
        const int counts_b = counts[cmap_b_idx];
        const float* peaks_a = &peaks[cmap_a_idx * M * 2];
        const float* peaks_b = &peaks[cmap_b_idx * M * 2];

        paf_score_graph_out_hw(score_graph_k, paf_i, paf_j, counts_a, counts_b, peaks_a, peaks_b, H, W, M,
                               num_integral_samples);
    }
}

void paf_score_graph_out_nkhw(float* score_graph,   // NxKxMxM
                              const int* topology,  // Kx4
                              const float* paf,     // Nx2KxHxW
                              const int* counts,    // NxC
                              const float* peaks,   // NxCxMx2
                              const int N, const int K, const int C, const int H, const int W, const int M,
                              const int num_integral_samples) {
    for (int n = 0; n < N; n++) {
        paf_score_graph_out_khw(&score_graph[n * K * M * M], topology, &paf[n * 2 * K * H * W], &counts[n * C],
                                &peaks[n * C * M * 2], K, C, H, W, M, num_integral_samples);
    }
}

class CoverTable {
public:
    CoverTable(int nrows, int ncols) : nrows(nrows), ncols(ncols) {
        rows.resize(nrows);
        cols.resize(ncols);
    }

    inline void coverRow(int row) {
        rows[row] = 1;
    }

    inline void coverCol(int col) {
        cols[col] = 1;
    }

    inline void uncoverRow(int row) {
        rows[row] = 0;
    }

    inline void uncoverCol(int col) {
        cols[col] = 0;
    }

    inline bool isCovered(int row, int col) const {
        return rows[row] || cols[col];
    }

    inline bool isRowCovered(int row) const {
        return rows[row];
    }

    inline bool isColCovered(int col) const {
        return cols[col];
    }

    inline void clear() {
        for (int i = 0; i < nrows; i++) {
            uncoverRow(i);
        }
        for (int j = 0; j < ncols; j++) {
            uncoverCol(j);
        }
    }

    const int nrows;
    const int ncols;

private:
    std::vector<bool> rows;
    std::vector<bool> cols;
};

class PairGraph {
public:
    PairGraph(int nrows, int ncols) : nrows(nrows), ncols(ncols) {
        this->rows.resize(nrows);
        this->cols.resize(ncols);
    }

    /**
     * Returns the column index of the pair matching this row
     */
    inline int colForRow(int row) const {
        return this->rows[row];
    }

    /**
     * Returns the row index of the pair matching this column
     */
    inline int rowForCol(int col) const {
        return this->cols[col];
    }

    /**
     * Creates a pair between row and col
     */
    inline void set(int row, int col) {
        this->rows[row] = col;
        this->cols[col] = row;
    }

    inline bool isRowSet(int row) const {
        return rows[row] >= 0;
    }

    inline bool isColSet(int col) const {
        return cols[col] >= 0;
    }

    inline bool isPair(int row, int col) {
        return rows[row] == col;
    }

    /**
     * Clears pair between row and col
     */
    inline void reset(int row, int col) {
        this->rows[row] = -1;
        this->cols[col] = -1;
    }

    /**
     * Clears all pairs in graph
     */
    void clear() {
        for (int i = 0; i < this->nrows; i++) {
            this->rows[i] = -1;
        }
        for (int j = 0; j < this->ncols; j++) {
            this->cols[j] = -1;
        }
    }

    int numPairs() {
        int count = 0;
        for (int i = 0; i < nrows; i++) {
            if (rows[i] >= 0) {
                count++;
            }
        }
        return count;
    }

    std::vector<std::pair<int, int>> pairs() {
        std::vector<std::pair<int, int>> p(numPairs());
        int count = 0;
        for (int i = 0; i < nrows; i++) {
            if (isRowSet(i)) {
                p[count++] = {i, colForRow(i)};
            }
        }
        return p;
    }

    const int nrows;
    const int ncols;

private:
    std::vector<int> rows;
    std::vector<int> cols;
};

void subMinRow(float* cost_graph, const int M, const int nrows, const int ncols) {
    for (int i = 0; i < nrows; i++) {
        // find min
        float min = cost_graph[i * M];
        for (int j = 0; j < ncols; j++) {
            float val = cost_graph[i * M + j];
            if (val < min) {
                min = val;
            }
        }

        // subtract min
        for (int j = 0; j < ncols; j++) {
            cost_graph[i * M + j] -= min;
        }
    }
}

void subMinCol(float* cost_graph, const int M, const int nrows, const int ncols) {
    for (int j = 0; j < ncols; j++) {
        // find min
        float min = cost_graph[j];
        for (int i = 0; i < nrows; i++) {
            float val = cost_graph[i * M + j];
            if (val < min) {
                min = val;
            }
        }

        // subtract min
        for (int i = 0; i < nrows; i++) {
            cost_graph[i * M + j] -= min;
        }
    }
}

void munkresStep1(const float* cost_graph, const int M, PairGraph& star_graph, const int nrows, const int ncols) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            if (!star_graph.isRowSet(i) && !star_graph.isColSet(j) && (cost_graph[i * M + j] == 0)) {
                star_graph.set(i, j);
            }
        }
    }
}

// returns 1 if we should exit
bool munkresStep2(const PairGraph& star_graph, CoverTable& cover_table) {
    int k = star_graph.nrows < star_graph.ncols ? star_graph.nrows : star_graph.ncols;
    int count = 0;
    for (int j = 0; j < star_graph.ncols; j++) {
        if (star_graph.isColSet(j)) {
            cover_table.coverCol(j);
            count++;
        }
    }
    return count >= k;
}

bool munkresStep3(const float* cost_graph, const int M, const PairGraph& star_graph, PairGraph& prime_graph,
                  CoverTable& cover_table, std::pair<int, int>& p, const int nrows, const int ncols) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            if (cost_graph[i * M + j] == 0 && !cover_table.isCovered(i, j)) {
                prime_graph.set(i, j);
                if (star_graph.isRowSet(i)) {
                    cover_table.coverRow(i);
                    cover_table.uncoverCol(star_graph.colForRow(i));
                } else {
                    p.first = i;
                    p.second = j;
                    return 1;
                }
            }
        }
    }
    return 0;
};

void munkresStep4(PairGraph& star_graph, PairGraph& prime_graph, CoverTable& cover_table, std::pair<int, int> p) {
    // repeat until no star found in prime's column
    while (star_graph.isColSet(p.second)) {
        // find and reset star in prime's column
        std::pair<int, int> s = {star_graph.rowForCol(p.second), p.second};
        star_graph.reset(s.first, s.second);

        // set this prime to a star
        star_graph.set(p.first, p.second);

        // repeat for prime in cleared star's row
        p = {s.first, prime_graph.colForRow(s.first)};
    }
    star_graph.set(p.first, p.second);
    cover_table.clear();
    prime_graph.clear();
}

void munkresStep5(float* cost_graph, const int M, const CoverTable& cover_table, const int nrows, const int ncols) {
    bool valid = false;
    float min;
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            if (!cover_table.isCovered(i, j)) {
                if (!valid) {
                    min = cost_graph[i * M + j];
                    valid = true;
                } else if (cost_graph[i * M + j] < min) {
                    min = cost_graph[i * M + j];
                }
            }
        }
    }

    for (int i = 0; i < nrows; i++) {
        if (cover_table.isRowCovered(i)) {
            for (int j = 0; j < ncols; j++) {
                cost_graph[i * M + j] += min;
            }
            //       cost_graph.addToRow(i, min);
        }
    }
    for (int j = 0; j < ncols; j++) {
        if (!cover_table.isColCovered(j)) {
            for (int i = 0; i < nrows; i++) {
                cost_graph[i * M + j] -= min;
            }
            //       cost_graph.addToCol(j, -min);
        }
    }
}

void _munkres(float* cost_graph, const int M, PairGraph& star_graph, const int nrows, const int ncols) {
    PairGraph prime_graph(nrows, ncols);
    CoverTable cover_table(nrows, ncols);
    prime_graph.clear();
    cover_table.clear();
    star_graph.clear();

    int step = 0;
    if (ncols >= nrows) {
        subMinRow(cost_graph, M, nrows, ncols);
    }
    if (ncols > nrows) {
        step = 1;
    }

    std::pair<int, int> p;
    bool done = false;
    while (!done) {
        switch (step) {
            case 0:
                subMinCol(cost_graph, M, nrows, ncols);
            case 1:
                munkresStep1(cost_graph, M, star_graph, nrows, ncols);
            case 2:
                if (munkresStep2(star_graph, cover_table)) {
                    done = true;
                    break;
                }
            case 3:
                if (!munkresStep3(cost_graph, M, star_graph, prime_graph, cover_table, p, nrows, ncols)) {
                    step = 5;
                    break;
                }
            case 4:
                munkresStep4(star_graph, prime_graph, cover_table, p);
                step = 2;
                break;
            case 5:
                munkresStep5(cost_graph, M, cover_table, nrows, ncols);
                step = 3;
                break;
        }
    }
}

std::size_t assignment_out_workspace(const int M) {
    return sizeof(float) * M * M;
}

void assignment_out(int* connections,          // 2xM
                    const float* score_graph,  // MxM
                    const int count_a, const int count_b, const int M, const float score_threshold, void* workspace) {
    const int nrows = count_a;
    const int ncols = count_b;

    // compute cost graph (negate score graph)
    float* cost_graph = (float*)workspace;
    for (int i = 0; i < count_a; i++) {
        for (int j = 0; j < count_b; j++) {
            const int idx = i * M + j;
            cost_graph[idx] = -score_graph[idx];
        }
    }

    // run munkres algorithm
    auto star_graph = PairGraph(nrows, ncols);
    _munkres(cost_graph, M, star_graph, nrows, ncols);

    // fill output connections
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            if (star_graph.isPair(i, j) && score_graph[i * M + j] > score_threshold) {
                connections[0 * M + i] = j;
                connections[1 * M + j] = i;
            }
        }
    }
}

void assignment_out_k(int* connections,          // Kx2xM
                      const float* score_graph,  // KxMxM
                      const int* topology,       // Kx4
                      const int* counts,         // C
                      const int K, const int M, const float score_threshold, void* workspace) {
    for (int k = 0; k < K; k++) {
        const int* tk = &topology[k * 4];
        const int cmap_idx_a = tk[2];
        const int cmap_idx_b = tk[3];
        const int count_a = counts[cmap_idx_a];
        const int count_b = counts[cmap_idx_b];
        assignment_out(&connections[k * 2 * M], &score_graph[k * M * M], count_a, count_b, M, score_threshold,
                       workspace);
    }
}

void assignment_out_nk(int* connections,          // NxKx2xM
                       const float* score_graph,  // NxKxMxM
                       const int* topology,       // Kx4
                       const int* counts,         // NxC
                       const int N, const int C, const int K, const int M, const float score_threshold,
                       void* workspace) {
    for (int n = 0; n < N; n++) {
        assignment_out_k(&connections[n * K * 2 * M], &score_graph[n * K * M * M], topology, &counts[n * C], K, M,
                         score_threshold, workspace);
    }
}

// 5. Merging
std::size_t connect_parts_out_workspace(const int C, const int M) {
    return sizeof(int) * C * M;
}

void connect_parts_out(int* object_counts,      // 1
                       int* objects,            // PxC
                       const int* connections,  // Kx2xM
                       const int* topology,     // Kx4
                       const int* counts,       // C
                       const int K, const int C, const int M, const int P, void* workspace) {
    // initialize objects
    for (int i = 0; i < C * M; i++) {
        objects[i] = -1;
    }

    // initialize visited
    std::memset(workspace, 0, connect_parts_out_workspace(C, M));
    int* visited = (int*)workspace;

    int num_objects = 0;

    for (int c = 0; c < C; c++) {
        if (num_objects >= P) {
            break;
        }

        const int count = counts[c];

        for (int i = 0; i < count; i++) {
            if (num_objects >= P) {
                break;
            }

            std::queue<std::pair<int, int>> q;
            bool new_object = false;
            q.push({c, i});

            while (!q.empty()) {
                auto node = q.front();
                q.pop();
                int c_n = node.first;
                int i_n = node.second;

                if (visited[c_n * M + i_n]) {
                    continue;
                }

                visited[c_n * M + i_n] = 1;
                new_object = true;
                objects[num_objects * C + c_n] = i_n;

                for (int k = 0; k < K; k++) {
                    const int* tk = &topology[k * 4];
                    const int c_a = tk[2];
                    const int c_b = tk[3];
                    const int* ck = &connections[k * 2 * M];

                    if (c_a == c_n) {
                        int i_b = ck[i_n];
                        if (i_b >= 0) {
                            q.push({c_b, i_b});
                        }
                    }

                    if (c_b == c_n) {
                        int i_a = ck[M + i_n];
                        if (i_a >= 0) {
                            q.push({c_a, i_a});
                        }
                    }
                }
            }

            if (new_object) {
                num_objects++;
            }
        }
    }
    *object_counts = num_objects;
}

void connect_parts_out_batch(int* object_counts,      // N
                             int* objects,            // NxPxC
                             const int* connections,  // NxKx2xM
                             const int* topology,     // Kx4
                             const int* counts,       // NxC
                             const int N, const int K, const int C, const int M, const int P, void* workspace) {
    for (int n = 0; n < N; n++) {
        connect_parts_out(&object_counts[n], &objects[n * P * C], &connections[n * K * 2 * M], topology, &counts[n * C],
                          K, C, M, P, workspace);
    }
}