#include "common_cuda.h"
#include "cuda.h"
#include "memory_manager.h"
#include "proto/onnx.pb.h"
#include <cassert>
#include <iostream>

using namespace onnx;
using namespace std;

void MemoryManager::register_placeholder(ValueInfoProto &tensor) {
  string name = tensor.name();
  if (contains(name)) {
    cerr << "Tensor " << name << " is already registered" << endl;
    return;
  }

  auto shape_info = tensor.type().tensor_type().shape();

  int num_elems = 1;
  for (auto d : shape_info.dim()) {
    num_elems *= d.dim_value();
  }
  //
  //  if (tensor.type().tensor_type().elem_type() != TensorProto_DataType_FLOAT)
  //  {
  //    cerr << "We expect float tensor only : " << endl;
  //    cerr << tensor.DebugString() << endl;
  //    assert(false);
  //  }

  CUdeviceptr ptr;
  CHECK_CUDEVICE(cuMemAlloc(&ptr, num_elems * sizeof(float)));

  storage.insert({name, make_tuple(ptr, num_elems)});
}

void MemoryManager::register_tensor(TensorProto &tensor) {
  string name = tensor.name();
  register_tensor(tensor, name);
}

void MemoryManager::register_tensor(TensorProto &tensor, string name) {
  FIJITTensor t = storage.at(name);

  CUdeviceptr ptr = get<0>(t);
  int num_elems = get<1>(t);

  CHECK_CUDEVICE(
      cuMemcpyHtoD(ptr, tensor.raw_data().c_str(), num_elems * sizeof(float)));
}

int MemoryManager::num_entries() { return storage.size(); }

bool MemoryManager::contains(string s) {
  return storage.find(s) != storage.end();
}

CUdeviceptr MemoryManager::get_device_ptr(string s) {
  return get<0>(storage.at(s));
}

float *MemoryManager::get_value(string s) {
  FIJITTensor t = storage.at(s);

  CUdeviceptr ptr = get<0>(t);
  int size = get<1>(t);

  float *arr = new float[size];
  CHECK_CUDEVICE(cuMemcpyDtoH(arr, ptr, size * sizeof(float)));
  return arr;
}