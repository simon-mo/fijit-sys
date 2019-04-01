//
// Created by Simon Mo on 2019-04-01.
//

#ifndef FIJIT_SYS_OPERATOR_H
#define FIJIT_SYS_OPERATOR_H

#include "cuda.h"
#include "model_manager.h"
#include "events.h"

#include "concurrentqueue/concurrentqueue.h"

#include <list>
#include <memory>
#include <tuple>

using namespace moodycamel;
using namespace std;

using PhysicalOpQueue = shared_ptr<ConcurrentQueue<shared_ptr<PhysicalOperator>>>;

struct ExecutorCtx {
  string model_name;
  int queue_id;
  cudaStream_t stream;
  PhysicalOpQueue queue;
};

class Executor {
 public:
  void register_queue(string model_name, int queue_id, PhysicalOpQueue queue_);
  void start();
  void stop();


 private:
  std::list<ExecutorCtx > executor_queues;
  bool should_stop = false;

  shared_ptr<EventRegistrar> events_registrar = get_gobal_event_registrar();
};

#endif //FIJIT_SYS_OPERATOR_H
