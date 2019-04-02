#include "abstract_operators.h"
#include "common_cuda.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cudnn.h"
#include "executor.h"
#include "memory_manager.h"
#include "model_manager.h"
#include "onnx_helper.h"
#include "operators.h"
#include "proto/onnx.pb.h"
#include "scheduler.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "google/protobuf/io/coded_stream.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "fmt/core.h"

#include "reporter.h"

#include "cxxopts/cxxopts.hpp"

using namespace std;
using namespace onnx;
using namespace google::protobuf::io;

void parse_model(ModelProto &model, const char *model_path) {
  int fd = open(model_path, O_RDONLY);
  ZeroCopyInputStream *raw_input = new FileInputStream(fd);
  CodedInputStream *coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(671088640, 167108860);

  model.ParseFromCodedStream(coded_input);

  close(fd);
}

void parse_input(TensorProto &input, string input_path) {
  fstream f(input_path, ios::in | ios::binary);
  input.ParseFromIstream(&f);
}

int main(int argc, char *argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  cxxopts::Options options("fijit-sys", "FIJIT Inference Engine");
  options.positional_help("[optional args]").show_positional_help();
  // clang-format off
  options.add_options()
      ("m,model", "Path to the model ONNX file", cxxopts::value<string>())
      ("i,input", "Path to the input ONNX file", cxxopts::value<string>())
      ("max-block", "Max block for TVM ops", cxxopts::value<int>())
      ("input-name", "Override input tensor name", cxxopts::value<string>())

      ("num-query", "Number of query per stream", cxxopts::value<int>()->default_value("1"))
      ("num-stream", "Number of stream", cxxopts::value<int>()->default_value("1"))

      ("h, help", "Print help");
  // clang-format on
  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help({"", "Group"}) << std::endl;
    exit(0);
  }

  const char *model_path = result["model"].as<string>().c_str();
  string input_path = result["input"].as<string>();
  int max_block = result["max-block"].as<int>();
  string input_name = result["input-name"].as<string>();
  int num_query = result["num-query"].as<int>();
  int num_stream = result["num-stream"].as<int>();

  ModelProto model;
  parse_model(model, model_path);

  TensorProto input;
  parse_input(input, input_path);

  CUcontext cudaCtx = cuda_init();
  cudnnHandle_t handle;
  cublasHandle_t cublasHandle;

  cudnnCreate(&handle);
  cublasCreate(&cublasHandle);

  //  cudaEvent_t start_of_world;
  //  CHECK_CUDA(cudaEventCreate(&start_of_world));
  //  CHECK_CUDA(cudaEventRecord(start_of_world, 0));

  shared_ptr<StaticMemoryManager> smm = make_shared<StaticMemoryManager>();
  shared_ptr<DynamicMemoryManager> dmm = make_shared<DynamicMemoryManager>();
  shared_ptr<ModelManager> model_manager = make_shared<ModelManager>(smm, dmm);

  string model_name = "default-model";
  model_manager->register_model(model, model_name);

  shared_ptr<ConcurrentQueue<shared_ptr<LogicalOperator>>> scheduler_queue =
      make_shared<ConcurrentQueue<shared_ptr<LogicalOperator>>>();

  auto scheduler = StaticScheduler(max_block, &cudaCtx, &handle, &cublasHandle);
  shared_ptr<ConcurrentQueue<shared_ptr<PhysicalOperator>>> dispatch_queue =
      scheduler.register_model_queue(model_name, scheduler_queue);
  thread scheduler_thread([&]() { scheduler.start(); });

  auto executor = Executor(&cudaCtx);
  executor.register_queue(model_name, dispatch_queue);
  thread executor_thread([&]() { executor.start(); });

  auto generate_query = [&](int query_id) {
    query_id = 0; // TODO(simon): we are overriding qid to re-use memory
    shared_ptr<vector<shared_ptr<LogicalOperator>>> ops =
        model_manager->instantiate_model(model_name, query_id);

    model_manager->register_input(model_name, query_id, input, input_name);

    for (auto o : *ops) {
      bool enqueue_result = scheduler_queue->enqueue(o);
      if (!enqueue_result) {
        cerr << "enqueued failed" << endl;
      }
    }
  };

  for (int j = 0; j < num_query; ++j) {
    for (int i = 0; i < num_stream; ++i) {
      generate_query(j * num_stream + i);
    }
  }

  this_thread::sleep_for(1s);

  CHECK_CUDA(cudaDeviceSynchronize());

  auto events =
      EventRegistrar::get_global_event_registrar().get_events(model_name);

  executor.stop();
  scheduler.stop();
  scheduler_thread.join();
  executor_thread.join();

  float total_time;
  cudaEventElapsedTime(&total_time, events[0][0],
                       events.at(events.size() - 1)[1]);
  cout << "Total time: " << total_time << " ms";
}
