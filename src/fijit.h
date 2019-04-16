//
// Created by Simon Mo on 2019-04-04.
//

#ifndef FIJIT_SYS_FIJIT_H
#define FIJIT_SYS_FIJIT_H

#include <map>
#include <memory>
#include <set>
#include <string>

#include "common/common_cuda.h"
#include "operators/abstract_operators.h"
#include "runtime/executor.h"
#include "runtime/memory_manager.h"
#include "runtime/model_manager.h"
#include "runtime/scheduler.h"
#include "utils/onnx_helper.h"

#include "concurrentqueue/concurrentqueue.h"

using namespace onnx;
using namespace std;

const set<string> ALLOWED_SCHEDULERS = {"StaticScheduler"};
typedef shared_ptr<ConcurrentQueue<vector<LogicalOperator>>> SchedQueue;
typedef shared_ptr<ConcurrentQueue<shared_ptr<PhysicalOperator>>> ExecQueue;

class Fijit {
public:
  Fijit();
  ~Fijit();

  // Add a model for profile
  void add_model(string path, string model_name,
                 vector<int> possible_blocks_config);

  // Add certain query load. Note that this is assume to be consistent across
  // configs
  void add_query(string path, string input_name);

  // Use scheduler
  void use_scheduler(string scheduler_name, map<string, int> config);

  // Use a static workload generator
  void use_workload(int qps, int total_query);

  // Prepare for inference benchmark, this will
  // - Load all operators
  // - Register all memories
  void prepare();

  // A blocking call for profiling.
  void infer();

private:
  shared_ptr<vector<LogicalOperator>> generate_query(string model_name,
                                                     int replica_id = 0);
  void wait_for_queues();

  map<string, ModelProto> model_protos;

  TensorProto input;
  string input_tensor_name;

  bool fijit_prepared = false;

  shared_ptr<StaticMemoryManager> smm = make_shared<StaticMemoryManager>();
  shared_ptr<DynamicMemoryManager> dmm = make_shared<DynamicMemoryManager>();
  shared_ptr<ModelManager> model_manager = make_shared<ModelManager>(smm, dmm);

  CUcontext cudaCtx;
  cudnnHandle_t handle;
  cublasHandle_t cublasHandle;

  thread scheduler_thread;
  thread executor_thread;

  int qps;
  int total_query;

  // shared_ptr<vector<LogicalOperator>> queries;
  map<string, shared_ptr<vector<LogicalOperator>>> model_to_queries;

  shared_ptr<Scheduler> scheduler;
  shared_ptr<Executor> executor;

  // shared_ptr<ConcurrentQueue<vector<LogicalOperator>>> scheduler_queue;
  // shared_ptr<ConcurrentQueue<shared_ptr<PhysicalOperator>>> dispatch_queue;
  map<string, SchedQueue> sched_queues;
  map<string, ExecQueue> exec_queues;
};

#endif // FIJIT_SYS_FIJIT_H
