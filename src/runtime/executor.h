//
// Created by Simon Mo on 2019-04-01.
//

#ifndef FIJIT_SYS_OPERATOR_H
#define FIJIT_SYS_OPERATOR_H

#include "runtime/common.h"
#include "common/common_cuda.h"
#include "runtime/events.h"
#include "runtime/model_manager.h"

#include <list>
#include <memory>
#include <tuple>
#include <vector>

using namespace moodycamel;
using namespace std;

class Executor {
public:
  Executor(cudaThreadContext ctx, InstQueue q, int num_streams);
  void start();
  void stop();

private:
  void wait();
  bool should_stop = false;

  cudaThreadContext ctx_;
  InstQueue q_;

  vector<cudaStream_t> streams;
  int num_streams_;

  EventRegistrar &events_registrar =
      EventRegistrar::get_global_event_registrar();
};

#endif // FIJIT_SYS_OPERATOR_H
