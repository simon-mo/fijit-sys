//
// Created by Simon Mo on 2019-03-14.
//

#ifndef FIJIT_SYS_ONNX_HELPER_H
#define FIJIT_SYS_ONNX_HELPER_H

#include "onnx.pb.h"
#include "onnx_helper.h"
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>

using namespace std;
using namespace onnx;

shared_ptr<unordered_map<string, ValueInfoProto>> traverse_shapes(GraphProto g);

size_t get_size(TensorProto &tensor);

size_t get_size(ValueInfoProto &info);

#endif // FIJIT_SYS_ONNX_HELPER_H
