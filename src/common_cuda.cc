//
// Created by Simon Mo on 2019-03-13.
//
#include "common_cuda.h"
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
