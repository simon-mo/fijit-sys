#include "fijit.h"
#include "utils/onnx_helper.h"

#include <algorithm>
#include <chrono>
#include <fmt/format.h>
#include <glog/logging.h>
#include <string>

using namespace std;
using namespace onnx;

void Fijit::add_model(string path, string model_name,
                      vector<int> possible_blocks_config) {
  ModelProto model;
  parse_model(model, path);

  CHECK(model_protos.find(model_name) == model_protos.end());
  LOG(INFO) << fmt::format("Adding model {}", model_name);
  model_protos.insert({model_name, model});
  model_manager->register_model(model, model_name, possible_blocks_config);
}

void Fijit::add_query(string path, string input_name) {
  parse_input(input, path);
  input_tensor_name = input_name;
}

void Fijit::use_scheduler(string scheduler_name, map<string, int> config) {
  int max_block = config.at("max_block");

  CHECK(ALLOWED_SCHEDULERS.find(scheduler_name) != ALLOWED_SCHEDULERS.end());
  // We only allows one type of scheuler now
  CHECK(scheduler_name == *ALLOWED_SCHEDULERS.begin());
  CHECK(scheduler_name == "StaticScheduler");

  scheduler =
      make_shared<StaticScheduler>(max_block, &cudaCtx, &handle, &cublasHandle);
  executor = make_shared<Executor>(&cudaCtx);
}

void Fijit::use_workload(int qps_, int total_query_) {
  qps = qps_;
  total_query = total_query_;
}

shared_ptr<vector<LogicalOperator>> Fijit::generate_query(string model_name,
                                                          int replica_id) {
  shared_ptr<vector<LogicalOperator>> ops =
      model_manager->instantiate_model(model_name, replica_id);

  auto all_blocks = model_manager->get_all_blocks_config(model_name);
  for (int i = 0; i < ops->size(); ++i) {
    ops->at(i).preload(all_blocks, &handle, &cublasHandle);
  }

  model_manager->register_input(model_name, replica_id, input,
                                input_tensor_name);

  return ops;
}

void Fijit::prepare() {
  fijit_prepared = true;

  // So far we support one model one replica and one input
  for (auto &kv : model_protos) {
    string model_name = kv.first;
    LOG(INFO) << fmt::format("Preparing input for model {}", model_name);

    auto scheduler_queue =
        make_shared<ConcurrentQueue<vector<LogicalOperator>>>();
    auto dispatch_queue =
        scheduler->register_model_queue(model_name, scheduler_queue);

    sched_queues.insert({model_name, scheduler_queue});
    exec_queues.insert({model_name, dispatch_queue});
    executor->register_queue(model_name, dispatch_queue);

    shared_ptr<vector<LogicalOperator>> queries = generate_query(model_name);
    model_to_queries.insert({model_name, queries});
  }

  scheduler_thread = thread([&]() { scheduler->start(); });
  executor_thread = thread([&]() { executor->start(); });
}

void Fijit::infer() {
  CHECK(fijit_prepared);

  auto sleep_time_ns =
      chrono::duration_cast<chrono::nanoseconds>(chrono::seconds(1)) / qps;

  for (size_t i = 0; i < total_query; i++) {
    auto start = chrono::high_resolution_clock::now();

    for (auto &kv : sched_queues) {
      auto queries = model_to_queries.at(kv.first);
      kv.second->enqueue(*queries);
    }

    auto end = chrono::high_resolution_clock::now();
    auto processing_time = end - start;
    auto time_left_to_sleep = sleep_time_ns - processing_time;

    // LOG(INFO) << fmt::format(
    //     "Sleeping for {}us",
    //     chrono::duration_cast<chrono::microseconds>(time_left_to_sleep)
    //         .count()
    //         );

    // CHECK(time_left_to_sleep > chrono::nanoseconds(1));
    if (time_left_to_sleep > chrono::nanoseconds(1)) {
      std::this_thread::sleep_for(time_left_to_sleep);
    }
  }

  std::this_thread::sleep_for(0.5s);
  wait_for_queues();

  CHECK_CUDA(cudaDeviceSynchronize());
}

// TODO(simon): Templatize this!
void wait_for_queue(SchedQueue q) {
  for (size_t i = 0; i < 2; i++) {
    while (q->size_approx() != 0) {
      std::this_thread::sleep_for(10ms);
    }
  }
}

void wait_for_queue(ExecQueue q) {
  for (size_t i = 0; i < 2; i++) {
    while (q->size_approx() != 0) {
      // LOG(INFO) << fmt::format("ExecQueue size {}", q->size_approx());
      std::this_thread::sleep_for(10ms);
    }
  }
}

void Fijit::wait_for_queues() {
  for (auto &kv : sched_queues) {
    LOG(INFO) << fmt::format("Waiting for sched queue {} to flush", kv.first);
    wait_for_queue(kv.second);
  }
  for (auto &kv : exec_queues) {
    LOG(INFO) << fmt::format("Waiting for exec queue {} to flush", kv.first);
    wait_for_queue(kv.second);
  }
}

Fijit::Fijit() {
  // Initialize CUDA Context
  cudaCtx = cuda_init();
  cudnnCreate(&handle);
  cublasCreate(&cublasHandle);
}

Fijit::~Fijit() {
  executor->stop();
  scheduler->stop();
  scheduler_thread.join();
  executor_thread.join();
}