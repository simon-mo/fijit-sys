//
// Created by Simon Mo on 2019-04-01.
//

#ifndef FIJIT_SYS_OPERATOR_H
#define FIJIT_SYS_OPERATOR_H

#include "common/common_cuda.h"
#include "runtime/events.h"
#include "runtime/model_manager.h"

#include "readerwriterqueue/readerwriterqueue.h"

#include <list>
#include <memory>
#include <tuple>
#include <vector>

using namespace moodycamel;
using namespace std;

using PhysicalOpQueue =
    shared_ptr<ReaderWriterQueue<shared_ptr<PhysicalOperator>>>;

struct ExecutorCtx {
  string model_name;
  cudaStream_t stream;
  PhysicalOpQueue queue;
};

class Executor {
public:
  Executor(CUcontext *ctx_) : ctx(ctx_){};
  void register_queue(string model_name, PhysicalOpQueue queue_);
  void start();
  void stop();

private:
  void wait();
  std::list<ExecutorCtx> executor_queues;
  bool should_stop = false;

  CUcontext *ctx;

  EventRegistrar &events_registrar =
      EventRegistrar::get_global_event_registrar();
};

#endif // FIJIT_SYS_OPERATOR_H
