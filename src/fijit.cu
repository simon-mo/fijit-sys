#include "fijit.h"
#include "onnx_helper.h"

#include <algorithm>
#include <chrono>
#include <fmt/format.h>
#include <glog/logging.h>
#include <string>

using namespace std;
using namespace onnx;

void Fijit::add_model(string path, int num_replica,
                      vector<int> possible_blocks_config) {
  ModelProto model;
  parse_model(model, path);

  CHECK(num_replica == 1);

  for (size_t i = 0; i < num_replica; i++) {
    model_protos.insert({path, model});
    model_manager->register_model(model, path, possible_blocks_config);
  }
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
  CHECK(model_protos.size() == 1);

  auto first_model_name = model_protos.begin()->first;

  scheduler_queue = make_shared<ConcurrentQueue<vector<LogicalOperator>>>();
  dispatch_queue =
      scheduler->register_model_queue(first_model_name, scheduler_queue);

  executor = make_shared<Executor>(&cudaCtx);
  executor->register_queue(first_model_name, dispatch_queue);

  scheduler_thread = thread([&]() { scheduler->start(); });
  executor_thread = thread([&]() { executor->start(); });

  queries = generate_query(first_model_name);
}

void Fijit::infer() {
  CHECK(fijit_prepared);

  auto sleep_time_ns =
      chrono::duration_cast<chrono::nanoseconds>(chrono::seconds(1)) / qps;

  for (size_t i = 0; i < total_query; i++) {
    auto start = chrono::high_resolution_clock::now();

    scheduler_queue->enqueue(*queries);
    auto end = chrono::high_resolution_clock::now();
    auto processing_time = end - start;
    auto time_left_to_sleep = sleep_time_ns - processing_time;
    CHECK(time_left_to_sleep > chrono::nanoseconds(1));

    LOG(INFO) << fmt::format(
        "Sleeping for {}us",
        chrono::duration_cast<chrono::microseconds>(time_left_to_sleep)
            .count());

    std::this_thread::sleep_for(time_left_to_sleep);
  }

  std::this_thread::sleep_for(0.5s);

  auto wait_for_queue_sched = [](decltype(scheduler_queue) q) {
    for (size_t i = 0; i < 2; i++) {
      while (q->size_approx() != 0) {
        std::this_thread::sleep_for(10ms);
      }
    }
  };
  wait_for_queue_sched(scheduler_queue);

  auto wait_for_queue_exec = [](decltype(dispatch_queue) q) {
    for (size_t i = 0; i < 2; i++) {
      while (q->size_approx() != 0) {
        std::this_thread::sleep_for(10ms);
      }
    }
  };
  wait_for_queue_exec(dispatch_queue);

  CHECK_CUDA(cudaDeviceSynchronize());
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