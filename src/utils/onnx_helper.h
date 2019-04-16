//
// Created by Simon Mo on 2019-03-14.
//

#ifndef FIJIT_SYS_ONNX_HELPER_H
#define FIJIT_SYS_ONNX_HELPER_H

#include "proto/onnx.pb.h"
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>

using namespace std;
using namespace onnx;

shared_ptr<unordered_map<string, ValueInfoProto>> traverse_shapes(GraphProto g);

size_t get_size(TensorProto &tensor);

size_t get_size(ValueInfoProto &info);

void parse_model(ModelProto &model, string model_path);

void parse_input(TensorProto &input, string input_path);

#endif // FIJIT_SYS_ONNX_HELPER_H
