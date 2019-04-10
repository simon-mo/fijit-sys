#include "memory_manager.h"
#include "model_manager.h"
#include "onnx_helper.h"
#include "operators.h"
#include "proto/onnx.pb.h"

#include <iostream>
#include <memory>
#include <set>
#include <unordered_map>

using namespace onnx;
using namespace std;

ModelManager::ModelManager(shared_ptr<StaticMemoryManager> smm,
                           shared_ptr<DynamicMemoryManager> dmm)
    : smm(smm), dmm(dmm) {}

void ModelManager::register_model(ModelProto model_proto, string model_name,
                                  vector<int> possible_blocks_config) {
  // If model_name is already registered, we just ignore the request.
  if (storage.find(model_name) != storage.end()) {
    return;
  }

  storage.insert({model_name, model_proto});
  possible_blocks_map.insert({model_name, possible_blocks_config});

  shape_maps.insert({model_name, traverse_shapes(model_proto.graph())});

  for (auto tensor : model_proto.graph().initializer()) {
    StaticBufferKey key{.model_name = model_name, .tensor_name = tensor.name()};
    smm->register_tensor(key, tensor);
  }
}

vector<int> ModelManager::get_all_blocks_config(string model_name) {
  return possible_blocks_map.at(model_name);
}

shared_ptr<vector<LogicalOperator>>
ModelManager::instantiate_model(string model_name, int query_id) {
  assert(storage.find(model_name) != storage.end());

  ModelProto proto = storage.at(model_name);
  shape_map_ptr shape = shape_maps.at(model_name);

  vector<LogicalOperator> op_queue;

  set<string> static_data;
  for (auto i : proto.graph().initializer()) {
    static_data.insert(i.name());
  }

  for (auto node : proto.graph().node()) {
    LogicalOperator op(node, shape);

    for (auto inp : node.input()) {

      // if this data is dynamic
      if (static_data.find(inp) == static_data.end()) {

        DynamicBufferKey key{
            .model_name = model_name, .tensor_name = inp, .query_id = query_id};

        // Register the memory
        dmm->register_placeholder(key, shape->at(inp));
        CUdeviceptr ptr = dmm->get_device_ptr(key);

        op.set_argument(INPUT, ptr);
      } else {
        StaticBufferKey key{.model_name = model_name, .tensor_name = inp};
        CUdeviceptr ptr = smm->get_device_ptr(key);
        op.set_argument(DATA, ptr);
      }
    }

    for (auto out : node.output()) {
      DynamicBufferKey key{
          .model_name = model_name, .tensor_name = out, .query_id = query_id};

      // Register the memory
      dmm->register_placeholder(key, shape->at(out));
      CUdeviceptr ptr = dmm->get_device_ptr(key);

      op.set_argument(OUTPUT, ptr);
    }

    op_queue.push_back(op);
  }

  return make_shared<decltype(op_queue)>(op_queue);
}

void ModelManager::register_input(string model_name, int query_id,
                                  TensorProto &tensor) {
  register_input(model_name, query_id, tensor, tensor.name());
}
void ModelManager::register_input(string model_name, int query_id,
                                  TensorProto &tensor, string tensor_name) {
  DynamicBufferKey key{
      .model_name = model_name,
      .tensor_name = tensor_name,
      .query_id = query_id,
  };
  dmm->register_tensor(key, tensor);
}