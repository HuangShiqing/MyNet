#include <map>
#include "tensor.h"
#include "base_infer.h"

enum class tensor_type {
    input,
    middle,
    output,
};

class Compare
{
private:
    /* data */
public:
    Compare(/* args */);
    ~Compare();
    void compare(std::map<std::string, Tensor> targets, std::map<std::string, Tensor> refs, float percent, tensor_type type);
    void compare_all(BaseInfer* target, BaseInfer* ref, float percent);
    void dump_result(std::string json_path);
    void dump_tensor(BaseInfer* target, BaseInfer* ref, std::string dir);
    std::map<std::string, std::vector<std::string>> result_outputs_;
    std::map<std::string, std::vector<std::string>> result_middles_;
    std::map<std::string, std::vector<std::string>> result_inputs_;
};
