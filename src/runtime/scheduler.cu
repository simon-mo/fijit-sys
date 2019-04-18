//
// Created by Simon Mo on 2019-03-21.
//
#include "runtime/scheduler.h"

#include "glog/logging.h"

using namespace std;
using namespace chrono;

void BaseScheduler::stop() { shouldStop = true; }

void StaticScheduler::start() {
  CHECK_CUDEVICE(cuCtxSetCurrent(*ctx_.cudaContext));

  while (true) {
    if (shouldStop) {
      break;
    }
    schedule();
  }
}

StaticScheduler::StaticScheduler(cudaThreadContext ctx, RequestQueue req_q,
                                 InstQueue inst_q, AlignmentDB &db,
                                 shared_ptr<ModelManager> mm)
    : ctx_{ctx}, req_q_{req_q}, inst_q_{inst_q}, db_{db}, mm_{mm} {};

void StaticScheduler::schedule() {
  RequestBatch batch;
  if (!(req_q_->try_dequeue(batch))){return;}
  CHECK(batch.size() > 0);

  vector<string> model_name_sorted;
  vector<int> batchsize;
  for (auto &kv : batch) {
    model_name_sorted.push_back(kv.first);
    batchsize.push_back(batch.count(kv.first));
  }

  vector<vector<LogicalOperator>> to_zip;

  // Create instruction queue in node order
  if (batch.size() == 1) {
    // Insert #batchsize quries
    for (int i = 0; i < batchsize[0]; ++i) {
      Request req = batch.erase(batch.begin())->second;
      vector<LogicalOperator> ops =
          mm_->instantiate_model(req.model_name, req.query_id);
      to_zip.push_back(move(ops));
    }

  } else {
    // Create instruction queue using aligned order
    AlignSolution sol = db_.get_align(model_name_sorted);
    for (auto model : model_name_sorted) {
      auto range = batch.equal_range(model);
      for (auto req = range.first; req != range.second; ++req) {
        Request &request = req->second;
        vector<LogicalOperator> ops = mm_->instantiate_model(
            request.model_name, request.query_id, sol.at(model));
        to_zip.push_back(move(ops));
      }
    }
  }

  // Check all queue has the same shape
  vector<int> op_queue_length;
  for (auto& v: to_zip) {
    op_queue_length.push_back(v.size());
  }
  for (auto& len: op_queue_length) {
    CHECK(len == op_queue_length[0]);
  }

  // Now ready to dispatch
  LogicalOperator begin(EventType::BEGIN);
  LogicalOperator end(EventType::END);
  
  inst_q_->enqueue({move(begin)});

  for(size_t i = 0; i < op_queue_length[0]; i++) {
    VLIW instruction;
    for(size_t q = 0; q < op_queue_length.size(); q++) {
      auto& model_q = to_zip.at(q);
      auto logical_op = model_q.erase(model_q.begin());
      if (logical_op->is_noop) {continue;}

      logical_op->preload(/*max blocks */ {}, ctx_.cudnnHandle, ctx_.cublasHandle);
      instruction.push_back(move(*logical_op));
    }
    inst_q_->enqueue(move(instruction));
  }

  inst_q_->enqueue({move(end)});
}