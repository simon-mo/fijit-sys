#ifndef FIJIT_SYS_MEMORY_MANAGER_H
#define FIJIT_SYS_MEMORY_MANAGER_H

#include "cuda.h"
#include "proto/onnx.pb.h"
#include <tuple>
#include <unordered_map>

using namespace std;
using namespace onnx;

typedef tuple<CUdeviceptr, int> FIJITTensor;

class MemoryManager {
public:
  void register_placeholder(ValueInfoProto &);
  void register_tensor(TensorProto &);
  void register_tensor(TensorProto &, string);
  int num_entries();
  CUdeviceptr get_device_ptr(string);
  float *get_value(string);

  bool contains(string);

private:
  unordered_map<string, FIJITTensor> storage;
};

#endif // FIJIT_SYS_MEMORY_MANAGER_H
