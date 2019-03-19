//
// Created by Simon Mo on 2019-03-16.
//
#include "common_cuda.h"
#include "cublas_v2.h"
#include "cublas_wrapper.h"

using namespace onnx;
using namespace std;

vector<cudaEvent_t> cuda_get_events_2(int num_events) {
  vector<cudaEvent_t> events(0);
  for (int i = 0; i < num_events; ++i) {
    cudaEvent_t e;
    CHECK_CUDA(cudaEventCreate(&e));
    events.push_back(e);
  }
  return events;
}

GemmOperator::GemmOperator(
    cublasHandle_t *handle_, NodeProto node,
    shared_ptr<unordered_map<string, ValueInfoProto>> io_shapes)
    : handle(handle_) {

  bool trans_A = false;
  bool trans_B = false;
  for (auto attri : node.attribute()) {
    if (attri.name() == "alpha") {
      alpha = attri.f();
    } else if (attri.name() == "beta") {
      beta = attri.f();
    } else if (attri.name() == "transA") {
      if (attri.i()) {
        //        transa = CUBLAS_OP_N;
        trans_A = true;
      }
    } else if (attri.name() == "transB") {
      if (attri.i()) {
        //        transb = CUBLAS_OP_N;
        trans_B = true;
      }
    }
  }

  ValueInfoProto A_info = io_shapes->at(node.input().Get(0));
  ValueInfoProto B_info = io_shapes->at(node.input().Get(1));
  ValueInfoProto C_info = io_shapes->at(node.output().Get(0));

  auto shape_vectors = [](ValueInfoProto p) {
    vector<int> shapes(0);
    for (auto d : p.type().tensor_type().shape().dim()) {
      shapes.push_back(d.dim_value());
    }
    return shapes;
  };

  vector<int> A_shape = shape_vectors(A_info);
  vector<int> B_shape = shape_vectors(B_info);
  vector<int> C_shape = shape_vectors(C_info);

  if (A_shape.size() == 4) {
    A_shape.pop_back();
    A_shape.pop_back();
  }


  assert((A_shape.size() == 2) && (B_shape.size() == 2) &&
         (C_shape.size() == 2));

  m = trans_A ? A_shape[1] : A_shape[0];
  k = trans_A ? A_shape[0] : A_shape[1];
  n = trans_B ? B_shape[0] : B_shape[1];

  assert(C_shape[0] == m);
  assert(C_shape[1] == n);

}

void GemmOperator::set_argument(KERNEL_ARG arg, CUdeviceptr ptr) {
  switch (arg) {
  case (INPUT):
    input = ptr;
    input_is_set = true;
    break;
  case (DATA):
    if (data_is_set) {
      bias = ptr;
      bias_is_set = true;
    } else {
      data = ptr;
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

vector<cudaEvent_t> GemmOperator::dispatch(cudaStream_t s) {
  assert(input_is_set && output_is_set && data_is_set && bias_is_set);

  vector<cudaEvent_t> events = cuda_get_events_2(2);
  CHECK_CUDA(cudaEventRecord(events[0], s));
  CHECK_CUBLAS(cublasSetStream(*handle, s));

  CHECK_CUDEVICE(cuMemcpyDtoDAsync(output, bias, m * n * sizeof(float), s));

  CHECK_CUBLAS(cublasSgemm(
      /* cublasHandle_t handle */ *handle,
      /* cublasOperation_t transa */ CUBLAS_OP_N,
      /* cublasOperation_t transb */ CUBLAS_OP_N,
      /* int m */ m,
      /* int n */ n,
      /* int k */ k,
      /* const float *alpha */ &alpha,
      /* const float *A */ (const float *)(input),
      /* int lda */ m,
      /* const float *B */ (const float *)(data),
      /* int ldb */ k, // TODO(simon) hard coded: assume B is transposed
      /* const float *beta */ &beta,
      /* float *C */ (float *)(output),
      /* int ldc */ m));

  CHECK_CUDA(cudaEventRecord(events[1], s));
  return events;
}

// TODO(simon) I had to patch cuda header, didn't have time to investigate why
/// usr/local/cuda/include/cuda_fp16.hpp(1680): error: more than one instance of
/// overloaded function "isinf" matches the argument list:
// function "isinf(float)"
// function "std::isinf(float)"
// argument types are: (float)
