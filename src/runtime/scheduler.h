//
// Created by Simon Mo on 2019-03-21.
//

#ifndef FIJIT_SYS_SCHEDULER_H
#define FIJIT_SYS_SCHEDULER_H

#include "runtime/model_manager.h"

#include "readerwriterqueue/readerwriterqueue.h"

#include <unordered_map>
#include <vector>

using namespace moodycamel;
using namespace std;

class Scheduler {
public:
  shared_ptr<ReaderWriterQueue<shared_ptr<PhysicalOperator>>>
  register_model_queue(
      string model_name,
      shared_ptr<ReaderWriterQueue<vector<LogicalOperator>>> q);

  void register_total_resource(shared_ptr<int> total_resource_estimate);

  virtual void schedule() = 0;

  virtual void start() = 0;

  void stop();

protected:
  bool shouldStop = false;

  unordered_map<string, shared_ptr<ReaderWriterQueue<vector<LogicalOperator>>>>
      logical_op_queues;
  unordered_map<string,
                shared_ptr<ReaderWriterQueue<shared_ptr<PhysicalOperator>>>>
      physical_op_queues;

  shared_ptr<int> total_resource = make_shared<int>(80);
};

class StaticScheduler : public Scheduler {
public:
  StaticScheduler(int max_blocks_per_model, CUcontext *ctx,
                  cudnnHandle_t *handle, cublasHandle_t *cublasHandle);
  void schedule() override;
  void start() override;

private:
  int max_blocks;
  cudnnHandle_t *handle;
  cublasHandle_t *cublasHandle;
  CUcontext *ctx;
};

#endif // FIJIT_SYS_SCHEDULER_H
