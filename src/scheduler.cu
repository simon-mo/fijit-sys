//
// Created by Simon Mo on 2019-03-21.
//

#include "scheduler.h"

#include "concurrentqueue/concurrentqueue.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

using namespace std;

shared_ptr<ConcurrentQueue<shared_ptr<PhysicalOperator>>>
Scheduler::register_model_queue(
    string model_name,
    shared_ptr<ConcurrentQueue<shared_ptr<LogicalOperator>>> q) {
  logical_op_queues.insert({model_name, q});

  shared_ptr<ConcurrentQueue<shared_ptr<PhysicalOperator>>> ops_q =
      make_shared<ConcurrentQueue<shared_ptr<PhysicalOperator>>>();
  physical_op_queues.insert({model_name, ops_q});

  return ops_q;
}

void Scheduler::register_total_resource(
    shared_ptr<int> total_resource_estimate) {
  total_resource = total_resource_estimate;
}

void Scheduler::stop() { shouldStop = true; }

void Scheduler::start() {
  while (true) {
    if (shouldStop) {
      break;
    }
    schedule();
    this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

StaticScheduler::StaticScheduler(int max_blocks_per_model,
                                 cudnnHandle_t *handle,
                                 cublasHandle_t *cublasHandle)
    : handle(handle), cublasHandle(cublasHandle) {
  max_blocks = max_blocks_per_model;
}

void StaticScheduler::schedule() {
  auto num_models = logical_op_queues.size();

  if (num_models * max_blocks > *total_resource) {
    cerr << "StaticScheduler::schedule allocated resource exceeds current "
            "total resource, skipping..."
         << endl;
    return;
  }

  for (auto &entry : logical_op_queues) {
    string model_name = entry.first;
    shared_ptr<ConcurrentQueue<shared_ptr<PhysicalOperator>>> dispatch_queue =
        physical_op_queues.at(model_name);

    shared_ptr<ConcurrentQueue<shared_ptr<LogicalOperator>>> op_queue =
        entry.second;

    shared_ptr<LogicalOperator> op;
    while (op_queue->try_dequeue(op)) {
      shared_ptr<PhysicalOperator> physical_op =
          op->realize(max_blocks, handle, cublasHandle);
      bool success = dispatch_queue->enqueue(physical_op);

      if (!success) {
        cerr << "Failed to enqueue operation to dispatch queue" << endl;
      }
    }
  }
}