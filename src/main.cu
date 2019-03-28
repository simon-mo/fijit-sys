#include "abstract_operators.h"
#include "common_cuda.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cudnn.h"
#include "memory_manager.h"
#include "model_manager.h"
#include "onnx_helper.h"
#include "operators.h"
#include "proto/onnx.pb.h"
#include "scheduler.h"
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

struct QueryContext {
  shared_ptr<vector<shared_ptr<LogicalOperator>>> logical_ops;
  shared_ptr<vector<shared_ptr<PhysicalOperator>>> physical_ops;
  shared_ptr<vector<vector<cudaEvent_t>>> events_collection;
};

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

  shared_ptr<StaticMemoryManager> smm = make_shared<StaticMemoryManager>();
  shared_ptr<DynamicMemoryManager> dmm = make_shared<DynamicMemoryManager>();
  shared_ptr<ModelManager> model_manager = make_shared<ModelManager>(smm, dmm);

  string model_name = "default-model";
  model_manager->register_model(model, model_name);

  auto scheduler_queue = make_shared<ConcurrentQueue<shared_ptr<LogicalOperator>>>();
  auto scheduler = StaticScheduler(max_block, &cudaCtx, &handle, &cublasHandle);
  auto dispatch_queue = scheduler.register_model_queue(model_name, scheduler_queue);
  thread scheduler_thread([&]() { scheduler.start(); });

  auto generate_query = [&](int query_id) {
    auto ops = model_manager->instantiate_model(model_name, query_id);
    model_manager->register_input(model_name, query_id, input, input_name);

    for (auto o : *ops) {
      bool enqueue_result = scheduler_queue->enqueue(o);
      if (!enqueue_result) {
        cerr << "enqueued failed" << endl;
      }
    }

    this_thread::sleep_for(std::chrono::milliseconds(150));

    // Step 4, realize operators
    auto physical_ops = make_shared<vector<shared_ptr<PhysicalOperator>>>();
    cerr << "Dispatching size " << dispatch_queue->size_approx() << endl;

    for (int i = 0; i < ops->size(); ++i) {
      shared_ptr<PhysicalOperator> op;
      bool deque_result = dispatch_queue->try_dequeue(op);
      if (!deque_result) {
        cerr << "can't deque operators, current physical ops size "
             << physical_ops->size() << " current dispatching size "
             << dispatch_queue->size_approx() << endl;
        this_thread::sleep_for(std::chrono::milliseconds(150));
      } else {
        physical_ops->push_back(op);
      }
    }

    auto events_collection = make_shared<vector<vector<cudaEvent_t>>>(0);

    QueryContext ctx{.logical_ops = ops,
                     .physical_ops = physical_ops,
                     .events_collection = events_collection};

    return ctx;
  };

  map<tuple<int, int>, QueryContext> dispatches;
  for (int j = 0; j < num_query; ++j) {
    for (int i = 0; i < num_stream; ++i) {
      QueryContext ctx = generate_query(j * num_stream + i);
      tuple<int, int> id = make_tuple(j, i);
      dispatches.insert({id, ctx});
    }
  }

  cudaEvent_t start_of_world;
  CHECK_CUDA(cudaEventCreate(&start_of_world));

  vector<cudaStream_t> streams;
  for (int k = 0; k < num_stream; ++k) {
    cudaStream_t s;
    CHECK_CUDA(cudaStreamCreate(&s));
    // CHECK_CUDA(cudaStreamWaitEvent(s, start_of_world, 0))
    streams.push_back(s);
  }

  CHECK_CUDA(cudaEventRecord(start_of_world, 0));

  for (int l = 0; l < num_query; ++l) {
    for (int i = 0; i < num_stream; ++i) {
      cudaStream_t s = streams[i];

      QueryContext ctx = dispatches.at(make_tuple(l, i));
      for (int k = 0; k < ctx.physical_ops->size(); ++k) {
        ctx.events_collection->emplace_back(ctx.physical_ops->at(k)->dispatch(s));
      }
    }
  }

  CHECK_CUDA(cudaDeviceSynchronize());


  ofstream trace;
  trace.open("out_trace.json");
  trace << "[";
  for (int m = 0; m < num_stream; ++m) {
    for (int q = 0; q < num_query; ++q) {
      QueryContext ctx = dispatches.at(make_tuple(q, m));
      TotalTimeReporter reporter_2(ctx.logical_ops, ctx.physical_ops, ctx.events_collection, start_of_world);

      float start;
      cudaEventElapsedTime(&start, start_of_world, ctx.events_collection->at(0)[0]);
      std::cout << start << std::endl;
      std::cout << reporter_2.report() << std::endl;

      ChromeTraceReporter reporter_1(ctx.logical_ops, ctx.physical_ops, ctx.events_collection, start_of_world);
      string trace_report = reporter_1.report(m, q);
      trace_report = trace_report.substr(1, trace_report.size() - 2);
      trace << trace_report << std::endl;
      if (q < num_query - 1 || m < num_stream - 1) {
        trace << ", ";
      }
    }
  }

  trace << "]";
  trace.close();
  scheduler.stop();
  scheduler_thread.join();
}
