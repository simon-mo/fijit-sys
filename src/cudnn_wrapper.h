//
// Created by Simon Mo on 2019-03-14.
//

#ifndef FIJIT_SYS_CUDNN_WRAPPER_H
#define FIJIT_SYS_CUDNN_WRAPPER_H

#include "../include/cuda.h"
#include "../include/cudnn.h"
#include "abstract_operators.h"
#include "common.h"
#include "cuda.h"
#include "cudnn.h"
#include "proto/onnx.pb.h"

#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace onnx;

class PoolingOperator : public CUDNNOperator {
public:
  PoolingOperator(cudnnHandle_t *handle_, ValueInfoProto input_shape_,
                  NodeProto node, cudnnPoolingMode_t mode);
  std::vector<cudaEvent_t> dispatch(cudaStream_t) override;
  void set_argument(KERNEL_ARG, CUdeviceptr) override;

  string get_name() { return "CudnnPooling";};

private:
  cudnnHandle_t *handle;
  CUdeviceptr input, output;
  bool input_is_set, output_is_set;
  cudnnPoolingDescriptor_t pool_desc;
  cudnnTensorDescriptor_t input_desc, output_desc;
};

class AddOperator : public CUDNNOperator {
public:
  AddOperator(cudnnHandle_t *handle_, ValueInfoProto input_shape_,
              ValueInfoProto output_shape_);
  std::vector<cudaEvent_t> dispatch(cudaStream_t) override;
  void set_argument(KERNEL_ARG, CUdeviceptr) override;

  string get_name() { return "CudnnAdd";};
private:
  cudnnHandle_t *handle;
  CUdeviceptr input, output, data;
  bool input_is_set, output_is_set, data_is_set;
  int total_size = 0;
  cudnnTensorDescriptor_t input_desc, output_desc;
};

class ReluOperator : public CUDNNOperator {
public:
  ReluOperator(cudnnHandle_t *handle_, ValueInfoProto input_shape_);
  std::vector<cudaEvent_t> dispatch(cudaStream_t) override;
  void set_argument(KERNEL_ARG, CUdeviceptr) override;

  string get_name() { return "CudnnRelu";};

private:
  cudnnHandle_t *handle;
  CUdeviceptr input, output;
  bool input_is_set, output_is_set;
  cudnnTensorDescriptor_t input_desc, output_desc;
  cudnnActivationDescriptor_t activation_desc;
};

class BatchNormOperator : public CUDNNOperator {
public:
  BatchNormOperator(cudnnHandle_t *handle_, ValueInfoProto input_shape_,
                    double epsilon);
  vector<cudaEvent_t> dispatch(cudaStream_t) override;
  void set_argument(KERNEL_ARG, CUdeviceptr) override;

  string get_name() { return "CudnnBatchNorm";};

private:
  cudnnHandle_t *handle;
  CUdeviceptr input, output;
  vector<CUdeviceptr> args;
  double epsilon;
  bool input_is_set, output_is_set, data_is_set;
  cudnnTensorDescriptor_t input_desc, output_desc, batch_norm_desc;
};

class SoftMaxOperator : public CUDNNOperator {
public:
  SoftMaxOperator(cudnnHandle_t *handle_, ValueInfoProto input_shape_);
  std::vector<cudaEvent_t> dispatch(cudaStream_t) override;
  void set_argument(KERNEL_ARG, CUdeviceptr) override;

  string get_name() { return "CudnnSoftMax";};

private:
  cudnnHandle_t *handle;
  CUdeviceptr input, output;
  bool input_is_set, output_is_set;
  cudnnTensorDescriptor_t input_desc, output_desc;
  cudnnActivationDescriptor_t activation_desc;

  cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_FAST;
  cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE;
};

class ConvOperator : public CUDNNOperator {
public:
  ConvOperator(cudnnHandle_t *handle_, NodeProto node,
               shared_ptr<unordered_map<string, ValueInfoProto>> io_shapes);
  std::vector<cudaEvent_t> dispatch(cudaStream_t) override;
  void set_argument(KERNEL_ARG, CUdeviceptr) override;

  string get_name() { return "CudnnConv";};

private:
  cudnnHandle_t *handle;
  CUdeviceptr input, output, data;
  bool input_is_set = false, output_is_set = false, data_is_set = false;

  cudnnTensorDescriptor_t input_descriptor;
  cudnnFilterDescriptor_t kernel_descriptor;
  cudnnConvolutionDescriptor_t convolution_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
};

#endif // FIJIT_SYS_CUDNN_WRAPPER_H
