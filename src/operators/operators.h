#ifndef OPERATORS_H
#define OPERATORS_H

#include "common/common.h"
#include "common/common_cuda.h"
#include "operators/abstract_operators.h"
#include "utils/onnx_helper.h"

#include <fmt/core.h>

#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

using namespace onnx;
using namespace std;

class TVMOperator : public PhysicalOperator {
public:
  TVMOperator(string, k_dim3, k_dim3, vector<KERNEL_ARG>, string);

  void set_argument(KERNEL_ARG, CUdeviceptr);

  void dispatch(cudaStream_t);

  bool operator==(const TVMOperator &rhs) const;

  CUfunction func;
  dim3 *block;
  dim3 *grid;
  vector<KERNEL_ARG> args;

  CUdeviceptr data, input, output;

  string get_name() { return "TVM-Conv"; };
  string get_name(int max_block) { return fmt::format("TVM-Conv-{}.", 42); };
  static string get_name_static(int max_block) {
    return fmt::format("TVM-Conv-{}.", 42);
  };

private:
  bool data_is_set = false, input_is_set = false, output_is_set = false;
};

class ReshapeOperator : public PhysicalOperator {
public:
  ReshapeOperator(int total_size);
  void set_argument(KERNEL_ARG arg, CUdeviceptr ptr);
  void dispatch(cudaStream_t stream);

  string get_name() { return "Native-Reshape"; };

private:
  int total_size;
  CUdeviceptr input, output;
  bool input_is_set = false, output_is_set = false;
};

class LogicalOperator {
public:
  LogicalOperator(bool is_noop);
  LogicalOperator(EventType event) : is_event(true), event_type(event) {};
  LogicalOperator(NodeProto, shared_ptr<unordered_map<string, ValueInfoProto>>);

  void preload(vector<int> possible_max_blocks, cudnnHandle_t *handle,
               cublasHandle_t *cublasHandle);

  shared_ptr<PhysicalOperator> realize(int max_blocks, cudnnHandle_t *handle,
                                       cublasHandle_t *cublasHandle);
  shared_ptr<TVMOperator> realize_tvm(int max_blocks);
  shared_ptr<CUBLASOperator> realize_cublas(cublasHandle_t *cublasHandle);
  shared_ptr<CUDNNOperator> realize_cudnn(cudnnHandle_t *handle);

  void set_argument(KERNEL_ARG k, CUdeviceptr ptr) {
    kernel_kwargs.push_back(make_tuple(k, ptr));
  }

  void inject_kwargs(shared_ptr<PhysicalOperator> op) {
    for (auto tup : kernel_kwargs) {
      KERNEL_ARG k;
      CUdeviceptr w;
      tie(k, w) = tup;
      op->set_argument(k, w);
    }
  }


  bool is_noop=false;

  bool is_event=false;
  EventType event_type;

private:
  string type;
  NodeProto node;
  shared_ptr<unordered_map<string, ValueInfoProto>> io_shapes;
  ValueInfoProto input_shape;
  ValueInfoProto output_shape;

  vector<tuple<KERNEL_ARG, CUdeviceptr>> kernel_kwargs;

  bool preloaded = false;
  map<int, shared_ptr<PhysicalOperator>> preloaded_ops;
  shared_ptr<PhysicalOperator> realize_preloaded(int max_blocks);
};

#endif /* OPERATORS_H */
