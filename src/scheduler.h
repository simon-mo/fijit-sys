//
// Created by Simon Mo on 2019-03-21.
//

#ifndef FIJIT_SYS_SCHEDULER_H
#define FIJIT_SYS_SCHEDULER_H

#include "model_manager.h"

#include "concurrentqueue/concurrentqueue.h"

#include <unordered_map>

using namespace moodycamel;

class Scheduler {
public:
  shared_ptr<ConcurrentQueue<PhysicalOperator>>
  register_model_queue(string model_name,
                       shared_ptr<ConcurrentQueue<LogicalOperator>> q);

  void register_total_resource(shared_ptr<int> total_resource_estimate);

  virtual void schedule() = 0;

private:
  unordered_map<string, shared_ptr<ConcurrentQueue<LogicalOperator>>>
      logical_op_queues;
  unordered_map<string, shared_ptr<ConcurrentQueue<LogicalOperator>>>
      physical_op_queues;
  shared_ptr<int> total_resource;
};

class StaticScheduler : public Scheduler {
public:
  StaticScheduler(int max_blocks_per_model);

private:
  int max_blocks;
};

#endif // FIJIT_SYS_SCHEDULER_H
