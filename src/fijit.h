//
// Created by Simon Mo on 2019-04-04.
//

#ifndef FIJIT_SYS_FIJIT_H
#define FIJIT_SYS_FIJIT_H

#include <map>
#include <memory>
#include <set>
#include <string>

#include "memory_manager.h"
#include "model_manager.h"
#include "onnx.pb.h"

#include "executor.h"
#include "scheduler.h"

#include "common_cuda.h"
#include "concurrentqueue/concurrentqueue.h"
#include "operators.h"

using namespace onnx;
using namespace std;

const set<string> ALLOWED_SCHEDULERS = {"StaticScheduler"};

class Fijit {
public:
  Fijit();

  void add_model(string path, int num_replica,
                 vector<int> possible_blocks_config);
  void add_query(string path, string input_name);

  void use_scheduler(string scheduler_name, map<string, int> config);

  void prepare();

  void infer();

private:
  map<string, ModelProto> model_protos;
  map<string, TensorProto> input_protos;

  shared_ptr<StaticMemoryManager> smm = make_shared<StaticMemoryManager>();
  shared_ptr<DynamicMemoryManager> dmm = make_shared<DynamicMemoryManager>();
  shared_ptr<ModelManager> model_manager = make_shared<ModelManager>(smm, dmm);

  CUcontext cudaCtx;
  cudnnHandle_t handle;
  cublasHandle_t cublasHandle;

  Scheduler scheduler;
  Executor executor;
  shared_ptr<ConcurrentQueue<vector<LogicalOperator>>> scheduler_queue;
  shared_ptr<ConcurrentQueue<shared_ptr<PhysicalOperator>>> dispatch_queue;
};

#endif // FIJIT_SYS_FIJIT_H
