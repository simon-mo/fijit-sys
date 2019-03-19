//
// Created by Simon Mo on 2019-03-14.
//

#ifndef FIJIT_SYS_ONNX_HELPER_H
#define FIJIT_SYS_ONNX_HELPER_H

#include "onnx.pb.h"
#include <memory>
#include <string>
#include <unordered_map>

using namespace std;

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

#endif // FIJIT_SYS_ONNX_HELPER_H
