#include "fijit.h"
#include "onnx_helper.h"

#include <string>

using namespace std;

void Fijit::add_model(string path, int num_replica,
                      vector<int> possible_blocks_config) {
  ModelProto model;
  parse_model(model, path);

  model_protos.insert({path, model});

  model_manager->register_model(model, path);
}

void Fijit::add_query(string path, string input_name) {
  TensorProto input;
  parse_input(input, path);

  input_protos.insert({path, input});
}

void Fijit::use_scheduler(string scheduler_name, map<int> config) {
  int max_block = config.at("max_block");
  assert(ALLOWED_SCHEDULERS.find(scheduler_name) != ALLOWED_SCHEDULERS.end());
  scheduler = StaticScheduler(max_block, &cudaCtx, &handle, &cublasHandle);
}

void Fijit::prepare() {
  assert(model_protos.size() == 1) assert(input_protos.size() == 1)

      scheduler_queue = make_shared<ConcurrentQueue<vector<LogicalOperator>>>();
  dispatch_queue =
      scheduler.register_model_queue(model_protos., scheduler_queue);
  executor = Executor(&cudaCtx);
  executor.register_queue(model_name, dispatch_queue);

  thread scheduler_thread([&]() { scheduler.start(); });
  thread executor_thread([&]() { executor.start(); });
}

Fijit::Fijit() {

  // Initialize CUDA Context
  cudaCtx = cuda_init();
  cudnnCreate(&handle);
  cublasCreate(&cublasHandle);

  auto generate_query = [&](int query_id) {
    query_id = 0; // TODO(simon): we are overriding qid to re-use memory
    shared_ptr<vector<LogicalOperator>> ops =
        model_manager->instantiate_model(model_name, query_id);

    for (int i = 0; i < ops->size(); ++i) {
      ops->at(i).preload(possible_blocks, &handle, &cublasHandle);
    }

    model_manager->register_input(model_name, query_id, input, input_name);

    return ops;
  };

  vector<shared_ptr<vector<LogicalOperator>>> queries;
  for (int j = 0; j < num_query; ++j) {
    for (int i = 0; i < num_stream; ++i) {
      queries.push_back(generate_query(j * num_stream + i));
    }
  }

  for (auto &ops : queries) {
    CHECK(scheduler_queue->enqueue(*ops));
  }

  this_thread::sleep_for(1s);

  CHECK_CUDA(cudaDeviceSynchronize());

  auto events =
      EventRegistrar::get_global_event_registrar().get_events(model_name);

  executor.stop();
  scheduler.stop();
  scheduler_thread.join();
  executor_thread.join();
}