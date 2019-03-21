//
// Created by Simon Mo on 2019-03-20.
//
#include "onnx_helper.h"

#include "onnx.pb.h"
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>

using namespace std;
using namespace onnx;

shared_ptr<unordered_map<string, ValueInfoProto>>
traverse_shapes(GraphProto g) {
  auto map = make_shared<unordered_map<string, ValueInfoProto>>();

  for (ValueInfoProto val_info : g.input()) {
    map->insert({val_info.name(), val_info});
  }

  for (ValueInfoProto val_info : g.output()) {
    map->insert({val_info.name(), val_info});
  }

  for (ValueInfoProto val_info : g.value_info()) {
    map->insert({val_info.name(), val_info});
  }

  return map;
}

size_t get_size(TensorProto &tensor) {
  auto shape_info = tensor.dims();
  size_t num_elems = std::accumulate(shape_info.begin(), shape_info.end(), 1,
                                     std::multiplies<size_t>());
  return num_elems;
}

size_t get_size(ValueInfoProto &info) {
  auto shape_info = info.type().tensor_type().shape().dim();
  size_t num_elems = 1;
  for (auto &shape : shape_info) {
    num_elems *= shape.dim_value();
  }
  return num_elems;
}
