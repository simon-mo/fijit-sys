#include "common/common_cuda.h"
#include "fijit.h"
#include "runtime/events.h"
#include "utils/backtrace.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>

#include "fmt/ostream.h"

#include "runtime/reporter.h"

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
      ("max-block", "Max block for TVM ops", cxxopts::value<string>())
      ("input-name", "Override input tensor name", cxxopts::value<string>())

      ("qps", "Query per seconds", cxxopts::value<string>()->default_value("10"))
      ("num-query", "Number of query per stream", cxxopts::value<string>()->default_value("1"))
      ("num-stream", "Number of stream", cxxopts::value<string>()->default_value("1"))

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

  int max_block = stoi(result["max-block"].as<string>());

  int num_query = stoi(result["num-query"].as<string>());
  int num_stream = stoi(result["num-stream"].as<string>());
  int qps = stoi(result["qps"].as<string>());

  vector<int> possible_blocks = {20, 40, 80, 120, 160, 200, 240, 320, 100000};
  CHECK(std::find(possible_blocks.begin(), possible_blocks.end(), max_block) !=
        possible_blocks.end());
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
