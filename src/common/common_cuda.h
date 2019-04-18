#ifndef FIJIT_SYS_COMMON_CUH_H
#define FIJIT_SYS_COMMON_CUH_H

#include "cublas_v2.h"
#include "cuda.h"
#include "cudnn.h"
#include <iostream>
#include <vector>

#define STR(x) #x

#define PRINT_ERR(expression, err_str)                                         \
  std::cerr << "Error on line " << STR(expression) << ": " << err_str << "\n"  \
            << __FILE__ << ":" << __LINE__ << std::endl;

#define CHECK_CUDA(expression)                                                 \
  {                                                                            \
    cudaError_t status = (expression);                                         \
    if (status != cudaSuccess) {                                               \
      const char *err_str = cudaGetErrorString(status);                        \
      PRINT_ERR(expression, err_str)                                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

#define CHECK_CUDEVICE(expression)                                             \
  {                                                                            \
    CUresult status = (expression);                                            \
    if (status != CUDA_SUCCESS) {                                              \
      const char *err_str;                                                     \
      cuGetErrorString(status, &err_str);                                      \
      PRINT_ERR(expression, err_str)                                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

#define CHECK_CUDNN(expression)                                                \
  {                                                                            \
    cudnnStatus_t status = (expression);                                       \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      const char *err_str;                                                     \
      err_str = cudnnGetErrorString(status);                                   \
      PRINT_ERR(expression, err_str)                                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

#define CHECK_CUBLAS(expression)                                               \
  {                                                                            \
    cublasStatus_t status = (expression);                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      PRINT_ERR(expression, status)                                            \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

#define CUDevicePtrConstCast(expression) (const void *)(uintptr_t)(expression)

#define CUDevicePtrCast(expression) (void *)(uintptr_t)(expression)

CUcontext cuda_init(void);

// This is suppose to be a stream callback that record a timestamp
// at memory location data
void CUDART_CB host_record_time(cudaStream_t stream, cudaError_t status,
                                void *data);

struct cudaThreadContext {
  CUcontext * cudaContext;
  cudnnHandle_t *cudnnHandle;
  cublasHandle_t *cublasHandle;
};

#endif // FIJIT_SYS_COMMON_CUH_H
