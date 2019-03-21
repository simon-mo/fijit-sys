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
#include <fstream>
#include <iostream>
#include <string>

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

  ModelProto model;
  parse_model(model, model_path);

  TensorProto input;
  parse_input(input, input_path);

  cuda_init();
  cudnnHandle_t handle;
  cublasHandle_t cublasHandle;

  cudnnCreate(&handle);
  cublasCreate(&cublasHandle);

  shared_ptr<StaticMemoryManager> smm = make_shared<StaticMemoryManager>();
  shared_ptr<DynamicMemoryManager> dmm = make_shared<DynamicMemoryManager>();
  shared_ptr<ModelManager> model_manager = make_shared<ModelManager>(smm, dmm);

  string model_name = "default-model";
  model_manager->register_model(model, model_name);

  int query_id = 0;
  vector<shared_ptr<LogicalOperator>> logical_ops =
      model_manager->instantiate_model(model_name, query_id);
  auto ops = make_shared<decltype(logical_ops)>(move(logical_ops));
  model_manager->register_input(model_name, query_id, input, input_name);

  // Step 4, realize operators
  shared_ptr<vector<shared_ptr<PhysicalOperator>>> physical_ops =
      make_shared<vector<shared_ptr<PhysicalOperator>>>();
  for (auto o : *ops) {
    physical_ops->push_back(o->realize(max_block, &handle, &cublasHandle));
  }

  // Step 6, start dispatch
  cudaStream_t s;
  cudaStreamCreate(&s);

  cudaEvent_t start_of_world;
  cudaEventCreate(&start_of_world);
  cudaEventRecord(start_of_world, s);

  shared_ptr<vector<vector<cudaEvent_t>>> events_collection =
      make_shared<vector<vector<cudaEvent_t>>>(0);

  for (int i = 0; i < physical_ops->size(); ++i) {
    events_collection->emplace_back(physical_ops->at(i)->dispatch(s));
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  // Step 7: Printout chrome://tracing data
  ChromeTraceReporter reporter_1(ops, physical_ops, events_collection,
                                 start_of_world);
  TotalTimeReporter reporter_2(ops, physical_ops, events_collection,
                               start_of_world);

  std::cout << reporter_1.report(1) << std::endl;
  std::cout << reporter_2.report() << std::endl;
}
