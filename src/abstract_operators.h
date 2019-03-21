//
// Created by Simon Mo on 2019-03-14.
//

#ifndef FIJIT_SYS_ABSTRACT_OPERATORS_H
#define FIJIT_SYS_ABSTRACT_OPERATORS_H

#include "common.h"
#include "cuda.h"
#include "cudnn.h"
#include <vector>

class PhysicalOperator {
public:
  virtual void set_argument(KERNEL_ARG, CUdeviceptr) = 0;

  virtual vector<cudaEvent_t> dispatch(cudaStream_t) = 0;

  virtual string get_name() = 0;
};

class CUDNNOperator : public PhysicalOperator {
public:
  vector<cudaEvent_t> events;
};

class CUBLASOperator : public PhysicalOperator {};

#endif // FIJIT_SYS_ABSTRACT_OPERATORS_H
