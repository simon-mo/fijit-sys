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
      ("i,input", "input config json", cxxopts::value<string>())
      ("n,num-trial", "number of trial", cxxopts::value<string>()->default_value("1"))
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

  string out_metric_path = result["out"].as<string>();
  int num_query = stoi(result["num-trial"].as<string>());
  string input_config_path = result["input"].as<string>();

  Fijit fijit;
  fijit.use_workload(input_config_path, num_query);
  fijit.infer();

  cudaDeviceSynchronize();

  auto events = EventRegistrar::get_global_event_registrar().get_events();

  LOG(INFO) << fmt::format("Total number of events {}", events.size());
  LOG(INFO) << fmt::format("First event {}", events.at(0));
  LOG(INFO) << fmt::format("Last event {}", events[events.size() - 1]);
  auto dur =
      chrono::nanoseconds(events[events.size() - 1].ts_ns - events[0].ts_ns);
  LOG(INFO) << fmt::format(
      "Duration {}us",
      chrono::duration_cast<chrono::microseconds>(dur).count());

  ofstream metric_file(out_metric_path);
  metric_file << report_chrome_trace();
}
