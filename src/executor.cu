#include "cuda.h"
#include "executor.h"

#include <thread>
#include <tuple>

void Executor::register_queue(string model_name, int queue_id,
                              PhysicalOpQueue queue) {
  cudaStream_t s;
  cudaStreamCreate(&s);

  ExecutorCtx ctx = {model_name, queue_id, s, queue};

  executor_queues.emplace_back(ctx);
}

void Executor::stop() { should_stop = true; }

void Executor::start() {
  while (true) {
    if (should_stop) {
      break;
    }

    shared_ptr<PhysicalOperator> op;
    for (ExecutorCtx &ctx_struct : executor_queues) {
      while (ctx_struct.queue->try_dequeue(op)) {
        auto events = op->dispatch(ctx_struct.stream);
        events_registrar->insert(ctx_struct.model_name, ctx_struct.queue_id,
                                 events);
      }
    }
  }
}