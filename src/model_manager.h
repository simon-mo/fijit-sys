//
// Created by Simon Mo on 2019-03-19.
//

#ifndef FIJIT_SYS_MODEL_MANAGER_H
#define FIJIT_SYS_MODEL_MANAGER_H

#include "memory_manager.h"
#include "onnx.pb.h"
#include "operators.h"

#include <memory>
#include <unordered_map>

using namespace onnx;
using namespace std;

typedef shared_ptr<unordered_map<string, ValueInfoProto>> shape_map_ptr;

class ModelManager {
public:
  ModelManager(shared_ptr<StaticMemoryManager>,
               shared_ptr<DynamicMemoryManager>);

  void register_model(ModelProto, string);

  shared_ptr<vector<LogicalOperator>> instantiate_model(string, int);

  void register_input(string, int, TensorProto &);
  void register_input(string, int, TensorProto &, string);

private:
  shared_ptr<StaticMemoryManager> smm;
  shared_ptr<DynamicMemoryManager> dmm;

  unordered_map<string, ModelProto> storage;
  unordered_map<string, shape_map_ptr> shape_maps;
};

#endif // FIJIT_SYS_MODEL_MANAGER_H
