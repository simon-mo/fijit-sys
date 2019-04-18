//
// Created by Simon Mo on 2019-03-21.
//

#ifndef FIJIT_SYS_SCHEDULER_H
#define FIJIT_SYS_SCHEDULER_H

#include "runtime/common.h"
#include "common/common_cuda.h"
#include "runtime/model_manager.h"
#include "utils/align_map.h"

#include "readerwriterqueue/readerwriterqueue.h"

#include <unordered_map>
#include <vector>

using namespace moodycamel;
using namespace std;

class BaseScheduler {
public:
  virtual void schedule() = 0;
  virtual void start() = 0;
  void stop();

protected:
  bool shouldStop = false;
};

class StaticScheduler : public BaseScheduler {
public:
  StaticScheduler(cudaThreadContext ctx, 
  RequestQueue req_q, InstQueue inst_q, shared_ptr<AlignmentDB> db, shared_ptr<ModelManager> mm);

  void schedule() override;
  void start() override;

private:
  cudaThreadContext ctx_;
  RequestQueue req_q_;
  InstQueue inst_q_;
  shared_ptr<AlignmentDB> db_;
  shared_ptr<ModelManager> mm_;
};

#endif // FIJIT_SYS_SCHEDULER_H
