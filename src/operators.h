#ifndef OPERATORS_H
#define OPERATORS_H

#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cudnn.h"

#include "proto/onnx.pb.h"

#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "abstract_operators.h"
#include "common.h"
#include "cudnn_wrapper.h"

#include <vector>

using namespace onnx;
using namespace std;

class TVMOperator : public PhysicalOperator {
public:
  TVMOperator(string, k_dim3, k_dim3, vector<KERNEL_ARG>, string);

  void set_argument(KERNEL_ARG, CUdeviceptr);

  vector<cudaEvent_t> dispatch(cudaStream_t);

  bool operator==(const TVMOperator &rhs) const;

  CUfunction func;
  dim3 *block;
  dim3 *grid;
  vector<KERNEL_ARG> args;

  CUdeviceptr data, input, output;

  string get_name() { return "TVM-Conv"; };

private:
  bool data_is_set = false, input_is_set = false, output_is_set = false;
};

class ReshapeOperator : public PhysicalOperator {
public:
  ReshapeOperator(int total_size);
  void set_argument(KERNEL_ARG arg, CUdeviceptr ptr);
  vector<cudaEvent_t> dispatch(cudaStream_t stream);

  string get_name() { return "Native-Reshape"; };

private:
  int total_size;
  CUdeviceptr input, output;
  bool input_is_set = false, output_is_set = false;
};

class LogicalOperator {
public:
  LogicalOperator(NodeProto, shared_ptr<unordered_map<string, ValueInfoProto>>);
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

private:
  string type;
  NodeProto node;
  shared_ptr<unordered_map<string, ValueInfoProto>> io_shapes;
  ValueInfoProto input_shape;
  ValueInfoProto output_shape;

  vector<tuple<KERNEL_ARG, CUdeviceptr>> kernel_kwargs;
};

#endif /* OPERATORS_H */
