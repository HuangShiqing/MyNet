#pragma once

#include <string>
#include <map>
#include "tensor.h"

class DataLoader
{
private:

public:
    DataLoader(std::string file_path, std::string file_type = "yaml");
    ~DataLoader();

    std::string file_path_;
    std::string file_type_;
    std::map<std::string, Tensor> input_tensors_;

    // void load_data();
};
