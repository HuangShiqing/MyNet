#pragma once

#include <string>
#include <map>
#include <vector>

enum class Dtype {
    fp32,
    int8
};

class Tensor
{
private:
    /* data */
public:
    Tensor(/* args */) = default;
    ~Tensor() = default;

    std::string name_;
    std::vector<int> shape_;
    std::vector<std::vector<int>> lod_;
    // std::vector<float> data_;//TODO:
    void* data_;
    int data_size_;
    Dtype dtype_;
};
