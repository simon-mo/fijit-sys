//
// Created by Simon Mo on 2019-03-16.
//

#ifndef FIJIT_SYS_CUBLAS_WRAPPER_H
#define FIJIT_SYS_CUBLAS_WRAPPER_H

#include "common/common.h"
#include "common/common_cuda.h"
#include "operators/abstract_operators.h"

#include "utils/onnx_helper.h"

#include <list>
#include <memory>
#include <vector>

using namespace onnx;
using namespace std;

class GemmOperator : public CUBLASOperator {
public:
  GemmOperator(cublasHandle_t *cublasHandle, NodeProto node,
               shared_ptr<unordered_map<string, ValueInfoProto>> io_shapes);
  void dispatch(cudaStream_t) override;
  void set_argument(KERNEL_ARG, CUdeviceptr) override;

  string get_name() { return "Cublas-Gemm"; };

private:
  cublasHandle_t *handle;
  CUdeviceptr input, output, data, bias;
  bool input_is_set = false, output_is_set = false, data_is_set = false,
       bias_is_set = false;

  cublasOperation_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_T;
  int m, n, k;
  float alpha = 1, beta = 1;
};

#endif // FIJIT_SYS_CUBLAS_WRAPPER_H
