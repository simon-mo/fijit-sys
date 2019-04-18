#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <tuple>


#include "proto/onnx.pb.h"

using namespace std;
using namespace onnx;

// FIJIT Type Definitions
enum KERNEL_ARG { 
  INPUT, 
  OUTPUT, 
  DATA 
};
typedef tuple<int, int, int> k_dim3;

// Event Type: Chrome Trace as a guide
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
enum class EventType : char {
  BEGIN = 'B',
  END = 'E',
};

std::ostream &operator<<(std::ostream &os, EventType t);

enum class EventSource : char {
  Scheduler = 'S',
  Executor = 'E',
  GPU = 'G',
};



#endif /* COMMON_H */
