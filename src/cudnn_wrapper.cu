#include "cuda.h"
#include "cudnn.h"

#include "common_cuda.h"
#include "cudnn_wrapper.h"

#include <cassert>
#include <vector>

using namespace onnx;
using namespace std;

vector<cudaEvent_t> cuda_get_events(int num_events) {
  vector<cudaEvent_t> events(0);
  for (int i = 0; i < num_events; ++i) {
    cudaEvent_t e;
    CHECK_CUDA(cudaEventCreate(&e));
    events.push_back(e);
  }
  return events;
}

PoolingOperator::PoolingOperator(cudnnHandle_t *handle_,
                                 ValueInfoProto input_shape_, NodeProto node,
                                 cudnnPoolingMode_t mode)
    : handle{handle_} {

  CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pool_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

  // events = cuda_get_events(2);

  vector<int> shapes(0);
  for (auto d : input_shape_.type().tensor_type().shape().dim()) {
    shapes.push_back(d.dim_value());
  }

  int kernel_h = 0, kernel_w = 0, v_pad = 0, h_pad = 0, v_stride = 0,
      h_stride = 0;
  for (auto attri : node.attribute()) {
    if (attri.name() == "kernel_shape") {
      kernel_h = attri.ints().Get(0);
      kernel_w = attri.ints().Get(1);
    }

    if (attri.name() == "pads") {
      v_pad = attri.ints().Get(2);
      h_pad = attri.ints().Get(3);
    }

    if (attri.name() == "strides") {
      v_stride = attri.ints().Get(0);
      h_stride = attri.ints().Get(1);
    }
  }

  if (kernel_h == 0 || kernel_w == 0) {
    kernel_h = shapes[2];
    kernel_w = shapes[3];
    v_pad = 0;
    h_pad = 0;
    v_stride = 1;
    h_stride = 1;
  }

  CHECK_CUDNN(cudnnSetPooling2dDescriptor(
      /* cudnnPoolingDescriptor_t poolingDesc */ pool_desc,
      /* cudnnPoolingMode_t mode */ mode,
      /* cudnnNanPropagation_t maxpoolingNanOpt */ CUDNN_PROPAGATE_NAN,
      /* int windowHeight */ kernel_h,
      /* int windowWidth */ kernel_w,
      /* int verticalPadding */ v_pad,
      /* int horizontalPadding */ h_pad,
      /* int verticalStride */ v_stride,
      /* int horizontalStride */ h_stride));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(
      /* cudnnTensorDescriptor_t tensorDesc */ input_desc,
      /* cudnnTensorFormat_t format */ CUDNN_TENSOR_NCHW,
      /* cudnnDataType_t dataType */ CUDNN_DATA_FLOAT,
      /* int n */ shapes[0],
      /* int c */ shapes[1],
      /* int h */ shapes[2],
      /* int w */ shapes[3]));

  int outN = 0, outC = 0, outH = 0, outW = 0;
  CHECK_CUDNN(cudnnGetPooling2dForwardOutputDim(
      /* const cudnnPoolingDescriptor_t poolingDesc */ pool_desc,
      /* const cudnnTensorDescriptor_t inputDesc */ input_desc,
      /* int *outN */ &outN,
      /* int *outC */ &outC,
      /* int *outH */ &outH,
      /* int *outW */ &outW));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(
      /* cudnnTensorDescriptor_t tensorDesc */ output_desc,
      /* cudnnTensorFormat_t format */ CUDNN_TENSOR_NCHW,
      /* cudnnDataType_t dataType */ CUDNN_DATA_FLOAT,
      /* int n */ outN,
      /* int c */ outC,
      /* int h */ outH,
      /* int w */ outW));
}

void PoolingOperator::dispatch(cudaStream_t s) {
  assert(input_is_set && output_is_set);

  float *scalers = new float[2];
  scalers[0] = 1;
  scalers[1] = 0;

  // CHECK_CUDA(cudaEventRecord(events[0], s));
  cudnnSetStream(*handle, s);
  CHECK_CUDNN(cudnnPoolingForward(
      /* cudnnHandle_t * handle */ *handle,
      /* const cudnnPoolingDescriptor_t poolingDesc */ pool_desc,
      /* const void *alpha */ scalers,
      /* const cudnnTensorDescriptor_t xDesc */ input_desc,
      /* const void *x */ (const void *)(uintptr_t)input,
      /* const void *beta */ scalers + 1,
      /* const cudnnTensorDescriptor_t yDesc */ output_desc,
      /* void *y */ (void *)(uintptr_t)output));
  // CHECK_CUDA(cudaEventRecord(events[1], s));
  // return events;
}

void PoolingOperator::set_argument(KERNEL_ARG arg, CUdeviceptr ptr) {

  switch (arg) {
  case (INPUT):
    input = ptr;
    input_is_set = true;
    break;
  case (OUTPUT):
    output = ptr;
    output_is_set = true;
    break;
  default:;
  }
}

AddOperator::AddOperator(cudnnHandle_t *handle_, ValueInfoProto input_shape_,
                         ValueInfoProto output_shape_)
    : handle{handle_} {
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
  // events = cuda_get_events(2);

  vector<int> input_shapes(0);
  for (auto d : input_shape_.type().tensor_type().shape().dim()) {
    input_shapes.push_back(d.dim_value());
  }

  vector<int> output_shapes(0);
  for (auto d : output_shape_.type().tensor_type().shape().dim()) {
    output_shapes.push_back(d.dim_value());
  }

  for (int i = 0; i < 4; ++i) {
    assert(input_shapes[i] == output_shapes[i]);
    total_size += input_shapes[i];
  }

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(
      /* cudnnTensorDescriptor_t tensorDesc */ input_desc,
      /* cudnnTensorFormat_t format */ CUDNN_TENSOR_NCHW,
      /* cudnnDataType_t dataType */ CUDNN_DATA_FLOAT,
      /* int n */ input_shapes[0],
      /* int c */ input_shapes[1],
      /* int h */ input_shapes[2],
      /* int w */ input_shapes[3]));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(
      /* cudnnTensorDescriptor_t tensorDesc */ output_desc,
      /* cudnnTensorFormat_t format */ CUDNN_TENSOR_NCHW,
      /* cudnnDataType_t dataType */ CUDNN_DATA_FLOAT,
      /* int n */ output_shapes[0],
      /* int c */ output_shapes[1],
      /* int h */ output_shapes[2],
      /* int w */ output_shapes[3]));
}

void AddOperator::set_argument(KERNEL_ARG arg, CUdeviceptr ptr) {
  if ((arg == INPUT) && (input_is_set)) {
    arg = DATA;
  }
  switch (arg) {
  case (INPUT):
    input = ptr;
    input_is_set = true;
    break;
  case (DATA):
    data = ptr;
    data_is_set = true;
    break;
  case (OUTPUT):
    output = ptr;
    output_is_set = true;
    break;
  default:;
  }
}

void AddOperator::dispatch(cudaStream_t s) {
  assert(input_is_set && output_is_set && data_is_set);

  float *scalers = new float[2];
  scalers[0] = 1;
  scalers[1] = 1;

  // CHECK_CUDA(cudaEventRecord(events[0], s));
  cudnnSetStream(*handle, s);

  cuMemcpyDtoDAsync(output, data, sizeof(float) * total_size, s);
  cudnnAddTensor(
      /* cudnnHandle_t handle */ *handle,
      /* *alpha */ scalers,
      /* aDesc */ input_desc,
      /* *A */ CUDevicePtrConstCast(input),
      /* *beta */ scalers + 1,
      /* cDesc */ output_desc,
      /* *C */ CUDevicePtrCast(output));

  // CHECK_CUDA(cudaEventRecord(events[1], s));
  // return events;
}

ReluOperator::ReluOperator(cudnnHandle_t *handle_, ValueInfoProto input_shape_)
    : handle{handle_} {
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
  // events = cuda_get_events(2);

  vector<int> input_shapes(0);
  for (auto d : input_shape_.type().tensor_type().shape().dim()) {
    input_shapes.push_back(d.dim_value());
  }

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(
      /* cudnnTensorDescriptor_t tensorDesc */ input_desc,
      /* cudnnTensorFormat_t format */ CUDNN_TENSOR_NCHW,
      /* cudnnDataType_t dataType */ CUDNN_DATA_FLOAT,
      /* int n */ input_shapes[0],
      /* int c */ input_shapes[1],
      /* int h */ input_shapes[2],
      /* int w */ input_shapes[3]));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(
      /* cudnnTensorDescriptor_t tensorDesc */ output_desc,
      /* cudnnTensorFormat_t format */ CUDNN_TENSOR_NCHW,
      /* cudnnDataType_t dataType */ CUDNN_DATA_FLOAT,
      /* int n */ input_shapes[0],
      /* int c */ input_shapes[1],
      /* int h */ input_shapes[2],
      /* int w */ input_shapes[3]));

  CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_desc));

  CHECK_CUDNN(cudnnSetActivationDescriptor(
      /* cudnnActivationDescriptor_t         activationDesc */ activation_desc,
      /* cudnnActivationMode_t               mode */ CUDNN_ACTIVATION_RELU,
      /* cudnnNanPropagation_t               reluNanOpt */ CUDNN_PROPAGATE_NAN,
      /* double                              coef */ 0.0));
}

void ReluOperator::set_argument(KERNEL_ARG arg, CUdeviceptr ptr) {
  switch (arg) {
  case (INPUT):
    input = ptr;
    input_is_set = true;
    break;
  case (OUTPUT):
    output = ptr;
    output_is_set = true;
    break;
  default:;
  }
}

void ReluOperator::dispatch(cudaStream_t s) {
  assert(input_is_set && output_is_set);

  float *scalers = new float[2];
  scalers[0] = 1;
  scalers[1] = 1;

  // CHECK_CUDA(cudaEventRecord(events[0], s));
  CHECK_CUDNN(cudnnSetStream(*handle, s));

  CHECK_CUDNN(cudnnActivationForward(
      /* cudnnHandle_t handle */ *handle,
      /* cudnnActivationDescriptor_t     activationDesc */ activation_desc,
      /* const void                     *alpha */ scalers,
      /* const cudnnTensorDescriptor_t   xDesc */ input_desc,
      /* const void                     *x */ CUDevicePtrConstCast(input),
      /* const void                     *beta */ scalers + 1,
      /* const cudnnTensorDescriptor_t   yDesc */ output_desc,
      /* void                           *y */ CUDevicePtrCast(output)));

  // CHECK_CUDA(cudaEventRecord(events[1], s));
  // return events;
}

BatchNormOperator::BatchNormOperator(cudnnHandle_t *handle_,
                                     ValueInfoProto input_shape_,
                                     double epsilon_)
    : handle(handle_), epsilon(epsilon_) {
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&batch_norm_desc));
  // events = cuda_get_events(2);

  vector<int> input_shapes(0);
  for (auto d : input_shape_.type().tensor_type().shape().dim()) {
    input_shapes.push_back(d.dim_value());
  }

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(
      /* cudnnTensorDescriptor_t tensorDesc */ input_desc,
      /* cudnnTensorFormat_t format */ CUDNN_TENSOR_NCHW,
      /* cudnnDataType_t dataType */ CUDNN_DATA_FLOAT,
      /* int n */ input_shapes[0],
      /* int c */ input_shapes[1],
      /* int h */ input_shapes[2],
      /* int w */ input_shapes[3]));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(
      /* cudnnTensorDescriptor_t tensorDesc */ output_desc,
      /* cudnnTensorFormat_t format */ CUDNN_TENSOR_NCHW,
      /* cudnnDataType_t dataType */ CUDNN_DATA_FLOAT,
      /* int n */ input_shapes[0],
      /* int c */ input_shapes[1],
      /* int h */ input_shapes[2],
      /* int w */ input_shapes[3]));

  CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(batch_norm_desc, input_desc,
                                            CUDNN_BATCHNORM_SPATIAL));
}

void BatchNormOperator::set_argument(KERNEL_ARG arg, CUdeviceptr ptr) {
  switch (arg) {
  case (INPUT):
    input = ptr;
    input_is_set = true;
    break;
  case (DATA):
    args.push_back(ptr);
    if (args.size() == 4) {
      data_is_set = true;
    }
    break;
  case (OUTPUT):
    output = ptr;
    output_is_set = true;
    break;
  default:;
  }
}

void BatchNormOperator::dispatch(cudaStream_t s) {
  assert(input_is_set && data_is_set && output_is_set);

  float *scalers = new float[2];
  scalers[0] = 1;
  scalers[1] = 0;

  // CHECK_CUDA(cudaEventRecord(events[0], s));
  CHECK_CUDNN(cudnnSetStream(*handle, s));

  if (epsilon < 1e-5) {
    epsilon += 1e-6; // handle the edge case where the value is exactly 1e-5
  }

  CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
      /* cudnnHandle_t handle */ *handle,
      /* cudnnBatchNormMode_t mode */ CUDNN_BATCHNORM_SPATIAL,
      /* const void *alpha */ scalers,
      /* const void *beta */ scalers + 1,
      /* const cudnnTensorDescriptor_t xDesc */ input_desc,
      /* const void *x */ CUDevicePtrConstCast(input),
      /* const cudnnTensorDescriptor_t yDesc */ output_desc,
      /* void *y */ CUDevicePtrCast(output),
      /* const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc */
      batch_norm_desc,
      /* const void *bnScale */ CUDevicePtrConstCast(args[0]),
      /* const void *bnBias */ CUDevicePtrConstCast(args[1]),
      /* const void *estimatedMean */ CUDevicePtrConstCast(args[2]),
      /* const void *estimatedVariance */ CUDevicePtrConstCast(args[3]),
      /* double epsilon */ epsilon));

  // CHECK_CUDA(cudaEventRecord(events[1], s));
  // return events;
}

SoftMaxOperator::SoftMaxOperator(cudnnHandle_t *handle_,
                                 ValueInfoProto input_shape_)
    : handle(handle_) {
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
  // events = cuda_get_events(2);

  vector<int> input_shapes(0);
  for (auto d : input_shape_.type().tensor_type().shape().dim()) {
    input_shapes.push_back(d.dim_value());
  }

  if (input_shapes.size() == 2) {
    input_shapes.push_back(1);
    input_shapes.push_back(1);
  }

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(
      /* cudnnTensorDescriptor_t tensorDesc */ input_desc,
      /* cudnnTensorFormat_t format */ CUDNN_TENSOR_NCHW,
      /* cudnnDataType_t dataType */ CUDNN_DATA_FLOAT,
      /* int n */ input_shapes[0],
      /* int c */ input_shapes[1],
      /* int h */ input_shapes[2],
      /* int w */ input_shapes[3]));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(
      /* cudnnTensorDescriptor_t tensorDesc */ output_desc,
      /* cudnnTensorFormat_t format */ CUDNN_TENSOR_NCHW,
      /* cudnnDataType_t dataType */ CUDNN_DATA_FLOAT,
      /* int n */ input_shapes[0],
      /* int c */ input_shapes[1],
      /* int h */ input_shapes[2],
      /* int w */ input_shapes[3]));
}

void SoftMaxOperator::set_argument(KERNEL_ARG arg, CUdeviceptr ptr) {
  switch (arg) {
  case (INPUT):
    input = ptr;
    input_is_set = true;
    break;
  case (OUTPUT):
    output = ptr;
    output_is_set = true;
    break;
  default:;
  }
}

void SoftMaxOperator::dispatch(cudaStream_t s) {
  assert(input_is_set && output_is_set);

  float *scalers = new float[2];
  scalers[0] = 1;
  scalers[1] = 0;

  // CHECK_CUDA(cudaEventRecord(events[0], s));
  CHECK_CUDNN(cudnnSetStream(*handle, s));

  CHECK_CUDNN(cudnnSoftmaxForward(
      /* cudnnHandle_t                    handle */ *handle,
      /* cudnnSoftmaxAlgorithm_t          algorithm */ algo,
      /* cudnnSoftmaxMode_t               mode */ mode,
      /* const void                      *alpha*/ scalers,
      /* const cudnnTensorDescriptor_t    xDesc*/ input_desc,
      /* const void                      *x*/ CUDevicePtrConstCast(input),
      /* const void                      *beta*/ scalers + 1,
      /* const cudnnTensorDescriptor_t    yDesc*/ output_desc,
      /* void                            *y*/ CUDevicePtrCast(output)));

  // CHECK_CUDA(cudaEventRecord(events[1], s));
  // return events;
}

ConvOperator::ConvOperator(
    cudnnHandle_t *handle_, NodeProto node,
    shared_ptr<unordered_map<string, ValueInfoProto>> io_shapes)
    : handle{handle_} {
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor));

  // events = cuda_get_events(2);

  string input_name = node.input().Get(0);
  string filter_name = node.input().Get(1);
  string output_name = node.output().Get(0);

  auto shape_vectors = [](ValueInfoProto p) {
    vector<int> shapes(0);
    for (auto d : p.type().tensor_type().shape().dim()) {
      shapes.push_back(d.dim_value());
    }
    return shapes;
  };

  auto input_shapes = shape_vectors(io_shapes->at(input_name));
  auto kernel_shapes = shape_vectors(io_shapes->at(filter_name));
  auto output_shapes = shape_vectors(io_shapes->at(output_name));

  auto attribute_vectors = [](NodeProto n, string attribute, int default_ = 0) {
    vector<int> values;
    for (auto attri : n.attribute()) {
      if (attri.name() == attribute) {
        for (auto val : attri.ints()) {
          values.push_back(val);
        }
      }
    }

    if (values.size() == 0) {
      values.push_back(default_);
      values.push_back(default_);
    }
    return values;
  };

  auto kernel_shape = attribute_vectors(node, "kernel_shape", 0);
  auto pads = attribute_vectors(node, "pads", 0);
  auto strides = attribute_vectors(node, "strides", 1);
  auto dilations = attribute_vectors(node, "dilations", 1);

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                         /*format=*/CUDNN_TENSOR_NCHW,
                                         /*dataType=*/CUDNN_DATA_FLOAT,
                                         /*batch_size=*/input_shapes[0],
                                         /*channels=*/input_shapes[1],
                                         /*image_height=*/input_shapes[2],
                                         /*image_width=*/input_shapes[3]));

  CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                         /*dataType=*/CUDNN_DATA_FLOAT,
                                         /*format=*/CUDNN_TENSOR_NCHW,
                                         /*out_channels=*/kernel_shapes[0],
                                         /*in_channels=*/kernel_shapes[1],
                                         /*kernel_height=*/kernel_shapes[2],
                                         /*kernel_width=*/kernel_shapes[3]));

  CHECK_CUDNN(
      cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                      /*pad_height=*/pads[0],
                                      /*pad_width=*/pads[1],
                                      /*vertical_stride=*/strides[0],
                                      /*horizontal_stride=*/strides[1],
                                      /*dilation_height=*/dilations[0],
                                      /*dilation_width=*/dilations[1],
                                      /*mode=*/CUDNN_CROSS_CORRELATION,
                                      /*computeType=*/CUDNN_DATA_FLOAT));

  int batch_size{0}, channels{0}, height{0}, width{0};
  CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
      convolution_descriptor, input_descriptor, kernel_descriptor, &batch_size,
      &channels, &height, &width));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                         /*format=*/CUDNN_TENSOR_NCHW,
                                         /*dataType=*/CUDNN_DATA_FLOAT,
                                         /*batch_size=*/batch_size,
                                         /*channels=*/channels,
                                         /*image_height=*/height,
                                         /*image_width=*/width));

  assert(batch_size == output_shapes[0]);
  assert(channels == output_shapes[1]);
  assert(height == output_shapes[2]);
  assert(width == output_shapes[3]);
}

void ConvOperator::set_argument(KERNEL_ARG arg, CUdeviceptr ptr) {
  switch (arg) {
  case (INPUT):
    input = ptr;
    input_is_set = true;
    break;
  case (DATA):
    if (data_is_set) {
      throw runtime_error("ConvOperator data set twice, maybe a bias term?");
    }
    data = ptr;
    data_is_set = true;
    break;
  case (OUTPUT):
    output = ptr;
    output_is_set = true;
    break;
  default:;
  }
}

void ConvOperator::dispatch(cudaStream_t s) {
  assert(input_is_set && output_is_set && data_is_set);

  float *scalers = new float[2];
  scalers[0] = 1;
  scalers[1] = 1;

  // CHECK_CUDA(cudaEventRecord(events[0], s));
  CHECK_CUDNN(cudnnSetStream(*handle, s));
  cudnnConvolutionForward(
      *handle, scalers, input_descriptor, CUDevicePtrConstCast(input),
      kernel_descriptor, CUDevicePtrConstCast(data), convolution_descriptor,
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, scalers + 1,
      output_descriptor, CUDevicePtrCast(output));

  // CHECK_CUDA(cudaEventRecord(events[1], s));
  // return events;
}