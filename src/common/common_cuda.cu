//
// Created by Simon Mo on 2019-03-13.
//
#include "common/common_cuda.h"
#include <chrono>
#include <vector>

using namespace std;

CUcontext cuda_init() {
  CUdevice cuDevice;
  CUcontext cuContext;
  CHECK_CUDEVICE(cuInit(0));
  CHECK_CUDEVICE(cuDeviceGet(&cuDevice, 0));
  CHECK_CUDEVICE(cuCtxCreate(&cuContext, 0, cuDevice));
  return cuContext;
}

void CUDART_CB host_record_time(cudaStream_t stream, cudaError_t status,
                                void *data) {
  int64_t *buf = (int64_t *)data;
  *buf = std::chrono::high_resolution_clock::now().time_since_epoch().count();
}
