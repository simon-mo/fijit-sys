#ifndef FIJIT_SYS_MEMORY_MANAGER_H
#define FIJIT_SYS_MEMORY_MANAGER_H

#include "cuda.h"
#include "proto/onnx.pb.h"
#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>

using namespace std;
using namespace onnx;

class StaticBufferKey {
public:
  string model_name;
  string tensor_name;

  //  StaticBufferKey(string m, string t) : model_name(m), tensor_name(t) {};

  bool operator==(const StaticBufferKey &k) const {
    return (model_name == k.model_name) && (tensor_name == k.tensor_name);
  };
};

class StaticBufferKeyHash {
public:
  std::size_t operator()(const StaticBufferKey &k) const {
    return hash<string>()(k.model_name) ^ hash<string>()(k.tensor_name);
  };
};

class DynamicBufferKey {
public:
  string model_name;
  string tensor_name;
  int query_id;

  bool operator==(const DynamicBufferKey &k) const {
    return (model_name == k.model_name) && (tensor_name == k.tensor_name) &&
           (query_id == k.query_id);
  };
  //  DynamicBufferKey(string m, string t, int id): model_name(m),
  //  tensor_name(t), query_id(id) {};
};

class DynamicBufferKeyHash {
public:
  std::size_t operator()(const DynamicBufferKey &k) const {
    return hash<string>()(k.model_name) ^ hash<string>()(k.tensor_name) ^
           hash<int>()(k.query_id);
  };
};

class BufferEntry {
public:
  CUdeviceptr ptr;
  size_t num_elems;

  //  BufferEntry(CUdeviceptr p, size_t num) : ptr(p), num_elems(num) {};
};

class StaticMemoryManager {
public:
  void register_tensor(StaticBufferKey &, TensorProto &);
  CUdeviceptr get_device_ptr(StaticBufferKey &);

private:
  unordered_map<StaticBufferKey, BufferEntry, StaticBufferKeyHash> storage;
};

class DynamicMemoryManager {
public:
  void register_placeholder(DynamicBufferKey &, ValueInfoProto &);
  void register_tensor(DynamicBufferKey &, TensorProto &);

  CUdeviceptr get_device_ptr(DynamicBufferKey &);

private:
  unordered_map<DynamicBufferKey, BufferEntry, DynamicBufferKeyHash> storage;
};

#endif // FIJIT_SYS_MEMORY_MANAGER_H
