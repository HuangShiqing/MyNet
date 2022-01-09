#include "yaml-cpp/yaml.h"
#include "data_loader.h"

DataLoader::DataLoader(std::string file_path, std::string file_type) {
    file_path_ = file_path;
    file_type_ = file_type;
        
    if (file_type_=="yaml") {
        // load yaml_input
        YAML::Node yaml_node = YAML::LoadFile(file_path_);
        for (auto it : yaml_node["input"]) {
            std::string name = it["name"].as<std::string>();
            std::vector<int> lod = it["lod"].as<std::vector<int>>();
            std::vector<int> shape = it["shape"].as<std::vector<int>>();
            std::vector<float> data = it["data"].as<std::vector<float>>();//TODO:

            Tensor t;
            t.name_ = name;
            t.shape_ = shape;
            t.lod_ = {lod};  // TODO:
            t.data_ = static_cast<void*>(data.data()); //TODO:
            t.dtype_ = Dtype::fp32;//TODO:
            input_tensors_.insert(std::make_pair(name, t));
        }
    } else {
        // TODO:
    }
};

DataLoader::~DataLoader()
{
}


