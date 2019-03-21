#include "common_cuda.h"
#include "cuda.h"
#include "memory_manager.h"
#include "onnx_helper.h"
#include "proto/onnx.pb.h"
#include <cassert>
#include <iostream>

using namespace onnx;
using namespace std;

void StaticMemoryManager::register_tensor(StaticBufferKey &key,
                                          TensorProto &tensor) {
  size_t num_elems = get_size(tensor);

  CUdeviceptr ptr;
  CHECK_CUDEVICE(cuMemAlloc(&ptr, num_elems * sizeof(float)));
  CHECK_CUDEVICE(
      cuMemcpyHtoD(ptr, tensor.raw_data().c_str(), num_elems * sizeof(float)));

  BufferEntry entry{.ptr = ptr, .num_elems = num_elems};

  storage.insert({key, entry});
}

CUdeviceptr StaticMemoryManager::get_device_ptr(StaticBufferKey &key) {
  return storage.at(key).ptr;
}

void DynamicMemoryManager::register_placeholder(DynamicBufferKey &key,
                                                ValueInfoProto &info) {
  size_t num_elems = get_size(info);

  CUdeviceptr ptr;
  CHECK_CUDEVICE(cuMemAlloc(&ptr, num_elems * sizeof(float)));

  BufferEntry entry{.ptr = ptr, .num_elems = num_elems};

  storage.insert({key, entry});
}

void DynamicMemoryManager::register_tensor(DynamicBufferKey &key,
                                           TensorProto &tensor) {
  size_t num_elems = get_size(tensor);
  BufferEntry entry = storage.at(key);

  assert(entry.num_elems == num_elems);
  CHECK_CUDEVICE(cuMemcpyHtoD(entry.ptr, tensor.raw_data().c_str(),
                              num_elems * sizeof(float)));
}

CUdeviceptr DynamicMemoryManager::get_device_ptr(DynamicBufferKey &key) {
  return storage.at(key).ptr;
}
