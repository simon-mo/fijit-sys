//
// Created by Simon Mo on 2019-03-21.
//

#include "common_cuda.h"
#include "scheduler.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

using namespace std;

shared_ptr<BlockingConcurrentQueue<shared_ptr<PhysicalOperator>>>
Scheduler::register_model_queue(
        string model_name,
shared_ptr<BlockingConcurrentQueue<shared_ptr<LogicalOperator>>> q) {
logical_op_queues.insert({model_name, q});

shared_ptr<BlockingConcurrentQueue<shared_ptr<PhysicalOperator>>> ops_q =
make_shared<BlockingConcurrentQueue<shared_ptr<PhysicalOperator>>>();

physical_op_queues.insert({model_name, ops_q});

return ops_q;
}

void Scheduler::register_total_resource(shared_ptr<int> total_resource_estimate) {
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
                                 cudnnHandle_t *handle_, cublasHandle_t *cublasHandle_)
        : max_blocks(max_blocks_per_model), ctx(ctx), handle(handle_), cublasHandle(cublasHandle_) {}

void StaticScheduler::schedule() {
    auto num_models = logical_op_queues.size();

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

    int num_entries = 0;
    for (auto &entry : logical_op_queues) {
        string model_name = entry.first;
        shared_ptr<BlockingConcurrentQueue<shared_ptr<PhysicalOperator>>> dispatch_queue =
                physical_op_queues.at(model_name);

        shared_ptr<BlockingConcurrentQueue<shared_ptr<LogicalOperator>>> op_queue =
                entry.second;

        // cerr << model_name << " queue size " << op_queue->size_approx() << endl;

        while (true) {
            shared_ptr<LogicalOperator> ops[1024];
            const int num_dequeued = op_queue->wait_dequeue_bulk_timed(ops, 1024, std::chrono::milliseconds(100));
            if (num_dequeued == 0)
                break;

            for (int i = 0; i < num_dequeued; i++) {
                shared_ptr<LogicalOperator> op = ops[i];
                shared_ptr<PhysicalOperator> physical_op = op->realize(max_blocks, handle, cublasHandle);
                bool success = dispatch_queue->enqueue(physical_op);

                if (!success) {
                    cerr << "Failed to enqueue operation to dispatch queue" << endl;
                }
                num_entries++;
            }
        }
    }

    // cerr << "\t[StaticScheduler] Loaded " << num_entries << " physical operations" << endl;
}
