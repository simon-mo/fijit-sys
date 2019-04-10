#include "common_cuda.h"
#include "cuda.h"
#include "executor.h"

#include <chrono>
#include <iostream>
#include <thread>
#include <tuple>

#include "fmt/core.h"
#include "glog/logging.h"

using namespace std;

void Executor::register_queue(string model_name, PhysicalOpQueue queue) {
  cudaStream_t s;
  cudaStreamCreate(&s);
  ExecutorCtx ctx = {model_name, s, queue};
  executor_queues.emplace_back(ctx);
}

void Executor::stop() {
  wait();
  should_stop = true;
}

void Executor::start() {
  CHECK_CUDEVICE(cuCtxSetCurrent(*ctx));

  while (true) {
    if (should_stop) {
      break;
    }

    shared_ptr<PhysicalOperator> op = nullptr;

    for (ExecutorCtx &ctx_struct : executor_queues) {
      while (ctx_struct.queue->try_dequeue(op)) {
        string op_name =
            fmt::format("{}-{}", ctx_struct.model_name, op->get_name());
        // events_registrar.record(EventType::BEGIN, EventSource::Executor,
        //                         op_name);
        events_registrar.record(EventType::BEGIN, EventSource::GPU, op_name,
                                ctx_struct.stream);
        auto events = op->dispatch(ctx_struct.stream);
        events_registrar.record(EventType::END, EventSource::GPU, op_name,
                                ctx_struct.stream);
        // events_registrar.record(EventType::END, EventSource::Executor,
        // op_name);
      }
    }
  }
}

void Executor::wait() {
  for (ExecutorCtx &ctx_struct : executor_queues) {
    while (ctx_struct.queue->size_approx() != 0) {
      std::this_thread::sleep_for(10ms);
    }
  }
  cudaDeviceSynchronize();
}