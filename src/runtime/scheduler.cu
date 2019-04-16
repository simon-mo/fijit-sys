//
// Created by Simon Mo on 2019-03-21.
//
#include "runtime/scheduler.h"

#include "common/common_cuda.h"

#include "operators/abstract_operators.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include <glog/logging.h>
#include "concurrentqueue/concurrentqueue.h"


using namespace std;
using namespace chrono;

shared_ptr<ConcurrentQueue<shared_ptr<PhysicalOperator>>>
Scheduler::register_model_queue(
    string model_name, shared_ptr<ConcurrentQueue<vector<LogicalOperator>>> q) {
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

void StaticScheduler::start() {
  CHECK_CUDEVICE(cuCtxSetCurrent(*ctx));

  while (true) {
    if (shouldStop) {
      break;
    }
    schedule();
  }
}

StaticScheduler::StaticScheduler(int max_blocks_per_model, CUcontext *ctx,
                                 cudnnHandle_t *handle_,
                                 cublasHandle_t *cublasHandle_)
    : max_blocks(max_blocks_per_model), ctx(ctx), handle(handle_),
      cublasHandle(cublasHandle_) {}

void StaticScheduler::schedule() {

  auto num_models = logical_op_queues.size();

  // if (num_models * max_blocks > *total_resource) {
  //   cerr << "StaticScheduler::schedule allocated resource exceeds current "
  //           "total resource, skipping..."
  //        << endl;
  //   return;
  // }

  for (auto &entry : logical_op_queues) {
    string model_name = entry.first;
    shared_ptr<ConcurrentQueue<shared_ptr<PhysicalOperator>>> dispatch_queue =
        physical_op_queues.at(model_name);

    shared_ptr<ConcurrentQueue<vector<LogicalOperator>>> op_queue =
        entry.second;

    vector<LogicalOperator> model_ops;
    // while (op_queue->try_dequeue(model_ops)) {
    if (!op_queue->try_dequeue(model_ops)) {
      continue;
    }
    shared_ptr<PhysicalOperator> begin_op = make_shared<TimingOperator>();
    shared_ptr<PhysicalOperator> end_op = make_shared<TimingOperator>();
    begin_op->is_timing = true;
    end_op->is_timing = true;
    begin_op->event_type = EventType::BEGIN;
    end_op->event_type = EventType::END;

    CHECK(dispatch_queue->enqueue(begin_op));
    for (auto &op : model_ops) {
      shared_ptr<PhysicalOperator> physical_op =
          op.realize(max_blocks, handle, cublasHandle);
      CHECK(dispatch_queue->enqueue(physical_op));
    }
    CHECK(dispatch_queue->enqueue(end_op));
    // }
  }
}