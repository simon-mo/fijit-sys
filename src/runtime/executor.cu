#include "runtime/executor.h"

#include "common/common.h"
#include "common/common_cuda.h"

#include <chrono>
#include <iostream>
#include <thread>
#include <tuple>

#include "fmt/core.h"
#include "glog/logging.h"

using namespace std;

Executor::Executor(cudaThreadContext ctx, InstQueue q, int num_streams): 
  ctx_(ctx), q_(q), num_streams_(num_streams) {
  streams.reserve(num_streams);
  
  for(size_t i = 0; i < num_streams; i++) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    streams.push_back(s);
  }
}

void Executor::stop() {
  wait();
  should_stop = true;
}

void Executor::start() {
  CHECK_CUDEVICE(cuCtxSetCurrent(*ctx_.cudaContext));
  VLIW inst;
  int cur_stream_id = 0;

  while (true) {
    if (should_stop) {
      break;
    }

    if (!(q_->try_dequeue(inst))) {continue;};

    CHECK(inst.size() <= num_streams_);

    //NOTE
    // Now we are at the point where the exectuor need to coalsce instruction
    // if necessary, but for now, let's make it work with basic time-multiplex
    // vector<cudaEvent_t> events;
    for (auto& op: inst) {
      if (op.is_event) {
        events_registrar.record(op.event_type, EventSource::Executor, "inst_queue", /*tid*/ 0);
      } else {
        shared_ptr<PhysicalOperator> kernel = op.realize(
          /*max_block*/0, ctx_.cudnnHandle, ctx_.cublasHandle);
        kernel->dispatch(streams[0]);
        // cur_stream_id ++;
      }
    }
  }
}

void Executor::wait() {
  while (q_->size_approx() != 0) {
    std::this_thread::sleep_for(10ms);
  }
  cudaDeviceSynchronize();
}