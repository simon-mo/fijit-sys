#ifndef FIJIT_SYS_ABSTRACT_OPERATORS_H
#define FIJIT_SYS_ABSTRACT_OPERATORS_H

#include "common/common.h"
#include "common/common_cuda.h"
#include <vector>

class PhysicalOperator {
public:
  virtual void set_argument(KERNEL_ARG, CUdeviceptr) = 0;

  virtual void dispatch(cudaStream_t) = 0;

  virtual string get_name() = 0;

  bool is_timing = false;
  EventType event_type;
};

class CUDNNOperator : public PhysicalOperator {};

class CUBLASOperator : public PhysicalOperator {};

class TimingOperator : public PhysicalOperator {
public:
  void set_argument(KERNEL_ARG, CUdeviceptr){};

  void dispatch(cudaStream_t){};

  string get_name() { return "TimingOp"; };

  bool is_timing = true;
  EventType event_type;
};

#endif // FIJIT_SYS_ABSTRACT_OPERATORS_H
