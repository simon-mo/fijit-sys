#include "abstract_operators.h"
#include "common_cuda.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cudnn.h"
#include "memory_manager.h"
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

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "cxxopts/cxxopts.hpp"

using namespace rapidjson;

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
  options.add_options()("m,model", "Path to the model ONNX file",
                        cxxopts::value<string>())(
      "i,input", "Path to the input ONNX file", cxxopts::value<string>())(
      "max-block", "Max block for TVM ops",
      cxxopts::value<int>())("input-name", "Override input tensor name",
                             cxxopts::value<string>())("h, help", "Print help");
  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help({"", "Group"}) << std::endl;
    exit(0);
  }

  const char *model_path = result["model"].as<string>().c_str();
  string input_path = result["input"].as<string>();
  int max_block = result["max-block"].as<int>();

  ModelProto model;
  parse_model(model, model_path);

  TensorProto input;
  parse_input(input, input_path);

  cuda_init();
  cudnnHandle_t handle;
  cudnnCreate(&handle);
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);

  // Step 1, malloc all data
  shared_ptr<MemoryManager> mm = make_shared<MemoryManager>();

  shared_ptr<unordered_map<string, ValueInfoProto>> shape_map =
      traverse_shapes(model.graph());

  for (auto name_to_val : *shape_map) {
    mm->register_placeholder(name_to_val.second);
  }

  // Step 2, fill in the placeholder
  for (auto tensor : model.graph().initializer()) {
    mm->register_tensor(tensor);
  }

  if (result.count("input-name")) {
    mm->register_tensor(input, result["input-name"].as<string>());
  } else {
    mm->register_tensor(input);
  }

  // Step 3, construct logical DAG
  vector<LogicalOperator> ops;
  for (auto n : model.graph().node()) {
    ops.emplace_back(n, shape_map);
  }

  // Step 4, realize operators
  vector<unique_ptr<PhysicalOperator>> physical_ops;
  for (auto o : ops) {
    physical_ops.push_back(o.realize(max_block, &handle, &cublasHandle));
  }

  // Step 5, connect the graph
  for (int i = 0; i < model.graph().node().size(); ++i) {
    auto n = model.graph().node().Get(i);

    physical_ops[i]->set_argument(INPUT, mm->get_device_ptr(n.input().Get(0)));

    if (n.input().size() > 1) {
      for (int j = 1; j < n.input().size(); ++j) {
        // TODO(simon): make sure we can set multiple data ptr and not overrides
        // then This is undefined for convs and pool
        physical_ops[i]->set_argument(DATA,
                                      mm->get_device_ptr(n.input().Get(j)));
      }
    }

    physical_ops[i]->set_argument(OUTPUT,
                                  mm->get_device_ptr(n.output().Get(0)));
  }

  // Step 6, start dispatch
  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaEvent_t start_of_world;
  cudaEventCreate(&start_of_world);
  cudaEventRecord(start_of_world, s);

  vector<vector<cudaEvent_t>> events_collection(0);
  for (auto &op : physical_ops) {
    vector<cudaEvent_t> events = op->dispatch(s);
    events_collection.push_back(events);
  }

  CHECK_CUDA(
      cudaEventSynchronize(events_collection[events_collection.size() - 1][1]));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Step 7: Printout chrome://tracing data

  //  [
  //  { "pid":1, "tid":1, "ts":87705, "dur":956189, "ph":"X", "name":"Jambase",
  //  "args":{ "ms":956.2 } },
  //  { "pid":1, "tid":1, "ts":128154, "dur":75867, "ph":"X",
  //  "name":"SyncTargets", "args":{ "ms":75.9 } },
  //  { "pid":1, "tid":1, "ts":546867, "dur":121564, "ph":"X",
  //  "name":"DoThings", "args":{ "ms":121.6 } }
  //  ]

  Document document;
  Document::AllocatorType &allocator = document.GetAllocator();

  document.SetArray();

  for (int k = 0; k < physical_ops.size(); ++k) {

    float duration, start_time;
    cudaEvent_t start = events_collection[k][0];
    cudaEvent_t end = events_collection[k][1];

    cudaEventElapsedTime(&duration, start, end);
    cudaEventElapsedTime(&start_time, start_of_world, start);

    string op_name = physical_ops[k]->get_name();
    string formatted_op_name = fmt::format("{}-{}", op_name, k);

    Value name;
    name.SetString(formatted_op_name.c_str(), formatted_op_name.size(),
                   allocator);

    Value item(kObjectType);
    item.AddMember("pid", Value(1), allocator);
    item.AddMember("tid", Value(1), allocator);
    item.AddMember("ts", start_time, allocator);
    item.AddMember("dur", duration, allocator);
    item.AddMember("ph", Value("X"), allocator);
    item.AddMember("name", name, allocator);
    document.PushBack(item.Move(), allocator);
  }

  StringBuffer buffer;
  Writer<StringBuffer> writer(buffer);
  document.Accept(writer);

  std::cout << buffer.GetString() << std::endl;

  // Step 8: Printout total latency

  float total_time;
  cudaEventElapsedTime(&total_time, start_of_world,
                       events_collection[events_collection.size() - 1][1]);
  std::cout << fmt::format("Total time: {} ms", total_time);
}
