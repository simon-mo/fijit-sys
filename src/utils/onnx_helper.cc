//
// Created by Simon Mo on 2019-03-20.
//
#include "utils/onnx_helper.h"

#include <fcntl.h>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

#include "google/protobuf/io/coded_stream.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>

using namespace std;
using namespace onnx;
using namespace google::protobuf::io;

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

void parse_model(ModelProto &model, string model_path) {
  int fd = open(model_path.c_str(), O_RDONLY);
  ZeroCopyInputStream *raw_input = new FileInputStream(fd);
  CodedInputStream *coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(671088640, 167108860);

  model.ParseFromCodedStream(coded_input);

  close(fd);
}

void parse_input(TensorProto &input, string input_path) {
  fstream f(input_path, ios::in | ios::binary);
  input.ParseFromIstream(&f);
}
