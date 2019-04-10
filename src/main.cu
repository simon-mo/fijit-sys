#include "backtrace.h"
#include "common_cuda.h"
#include "events.h"
#include "fijit.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>

#include "fmt/ostream.h"

#include "reporter.h"

#include "cxxopts/cxxopts.hpp"
#include <glog/logging.h>

using namespace std;
using namespace onnx;

using namespace chrono;

//
//  cudaStream_t s;
//  cudaStreamCreate(&s);
//  CHECK_CUDA(cudaStreamAddCallback(s, print_time, nullptr, 0));

int main(int argc, char *argv[]) {
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  google::InitGoogleLogging(argv[0]);

  GOOGLE_PROTOBUF_VERIFY_VERSION;

  cxxopts::Options options("fijit-sys", "FIJIT Inference Engine");
  options.positional_help("[optional args]").show_positional_help();
  // clang-format off
  options.add_options()
      ("m,model", "Path to the model ONNX file", cxxopts::value<string>())
      ("i,input", "Path to the input ONNX file", cxxopts::value<string>())
      ("max-block", "Max block for TVM ops", cxxopts::value<int>())
      ("input-name", "Override input tensor name", cxxopts::value<string>())

      ("qps", "Query per seconds", cxxopts::value<int>()->default_value("10"))
      ("num-query", "Number of query per stream", cxxopts::value<int>()->default_value("1"))
      ("num-stream", "Number of stream", cxxopts::value<int>()->default_value("1"))

      ("out", "Metric file path", cxxopts::value<string>())

      ("backtrace", "Print backtrace on crash")

      ("h, help", "Print help");
  // clang-format on
  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help({"", "Group"}) << std::endl;
    exit(0);
  }

  if (result.count("backtrace")) {
    std::set_terminate([]() { backtrace(); });
  }

  string model_path = result["model"].as<string>();

  string input_path = result["input"].as<string>();
  string input_name = result["input-name"].as<string>();

  string out_metric_path = result["out"].as<string>();

  int max_block = result["max-block"].as<int>();

  int num_query = result["num-query"].as<int>();
  int num_stream = result["num-stream"].as<int>();
  int qps = result["qps"].as<int>();

  vector<int> possible_blocks = {20, 40, 80};
  map<string, int> sched_config = {{"max_block", max_block}};

  Fijit fijit;

  for (size_t i = 0; i < num_stream; i++) {
    fijit.add_model(model_path, fmt::format("model-{}", i), possible_blocks);
  }

  fijit.add_query(input_path, input_name);
  fijit.use_scheduler("StaticScheduler", sched_config);
  fijit.use_workload(qps, num_query);

  fijit.prepare();
  fijit.infer();

  auto events = EventRegistrar::get_global_event_registrar().get_events();

  LOG(INFO) << fmt::format("Total number of events {}", events.size());
  LOG(INFO) << fmt::format("First event {}", events[0]);
  LOG(INFO) << fmt::format("Last event {}", events[events.size() - 1]);
  auto dur =
      chrono::nanoseconds(events[events.size() - 1].ts_ns - events[0].ts_ns);
  LOG(INFO) << fmt::format(
      "Duration {}us",
      chrono::duration_cast<chrono::microseconds>(dur).count());

  ofstream metric_file(out_metric_path);
  metric_file << report_chrome_trace();
}
