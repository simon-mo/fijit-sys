#include "common_cuda.h"
#include "cuda.h"
#include "executor.h"

#include <chrono>
#include <iostream>
#include <thread>
#include <tuple>

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
        op->dispatch(ctx_struct.stream);
        // events_registrar.insert(ctx_struct.model_name, events);
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