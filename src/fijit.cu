#include "fijit.h"
#include "utils/onnx_helper.h"

#include <algorithm>
#include <chrono>
#include <fmt/format.h>
#include <glog/logging.h>
#include <string>
#include <thread>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/stringbuffer.h"
#include <cstdio>
#include <rapidjson/writer.h>

#include "operators/cudnn_wrapper.h"
#include "operators/operators.h"

using namespace rapidjson;
using namespace std;
using namespace onnx;

enum class Strategy { COALESCE = 'C', TIME_MULTIPLEX = 'T' };

void Fijit::add_model(string path, string model_name,
                      vector<int> possible_blocks_config) {
  ModelProto model;
  parse_model(model, path);

  CHECK(model_protos.find(model_name) == model_protos.end());
  LOG(INFO) << fmt::format("Adding model {}", model_name);
  model_protos.insert({model_name, model});
  model_manager->register_model(model, model_name, possible_blocks_config);

  model_instantiated.insert(
      {model_name, model_manager->instantiate_model(model_name, /*req id*/ 0)});

  cudaStream_t s;
  cudaStreamCreate(&s);
  model_streams.insert({model_name, s});
}

void Fijit::add_query(string model_name, string path, string input_name) {
  TensorProto input;
  parse_input(input, path);

  input_protos.emplace(model_name, move(input));
  input_tensor_names.emplace(model_name, input_name);
}

// void Fijit::use_scheduler(string scheduler_name, map<string, int> config) {
//   int max_block = config.at("max_block");

//   CHECK(ALLOWED_SCHEDULERS.find(scheduler_name) != ALLOWED_SCHEDULERS.end());
//   // We only allows one type of scheuler now
//   CHECK(scheduler_name == *ALLOWED_SCHEDULERS.begin());
//   CHECK(scheduler_name == "StaticScheduler");

//   scheduler =
//       make_shared<StaticScheduler>(max_block, &cudaCtx, &handle,
//       &cublasHandle);
//   executor = make_shared<Executor>(&cudaCtx);
// }

Document parse_json(string path) {
  FILE *fp = fopen(path.c_str(), "r");
  char readBuffer[65536];
  FileReadStream is(fp, readBuffer, sizeof(readBuffer));
  Document d;
  d.ParseStream(is);
  fclose(fp);
  return d;
}

void Fijit::register_workload(Document &doc) {
  const Value &models = doc["models"];
  CHECK(models.IsArray());
  for (SizeType i = 0; i < models.Size(); i++) {
    string model_name = models[i]["model_name"].GetString();
    string model_path = models[i]["model_path"].GetString();
    string input_name = models[i]["input_name"].GetString();
    string input_path = models[i]["input_path"].GetString();

    add_model(model_path, model_name, {20, 40, 80, 160});
    add_query(model_name, input_path, input_name);
  }
}

void Fijit::use_workload(string path, int num_queries) {
  total_query = num_queries;

  Document op_data = parse_json(path);
  register_workload(op_data);

  const Value &schedule = op_data["schedule"];
  CHECK(schedule.IsArray());
  for (SizeType i = 0; i < schedule.Size(); i++) {
    // {
    //   "ops": [
    //     {
    //       "model_name": "resnet50_v1",
    //       "node_id": 0
    //     },
    //     {
    //       "model_name": "resnet50_v1",
    //       "node_id": 0
    //     }
    //   ],
    //   "strategy": "Coalsce"
    // }
    string strategy = schedule[i]["strategy"].GetString();
    char strategy_code = strategy.c_str()[0];
    if (strategy_code == static_cast<char>(Strategy::COALESCE)) {

      auto &ops = schedule[i]["ops"];
      CUdeviceptr ptrs[ops.Size()];
      for (SizeType j = 0; j < ops.Size(); j++) {
        auto model_name = ops[j]["model_name"].GetString();
        int node_id = ops[j]["node_id"].GetInt();
        const NodeProto &node =
            model_protos.at(model_name).graph().node(node_id);

        auto im2col = make_shared<Im2ColOperator>(
            &handle, node, model_manager->get_shape_map(model_name));

        CUdeviceptr ptr;
        LOG(INFO) << "Malloc " << im2col->output_buffer_size * sizeof(float)
                  << " floats";
        CHECK_CUDEVICE(
            cuMemAlloc(&ptr, im2col->output_buffer_size * sizeof(float)));
        ptrs[j] = ptr;

        model_instantiated.at(model_name)->at(node_id).inject_kwargs(im2col);
        im2col->set_argument(KERNEL_ARG::OUTPUT, ptr);

        op_queue.push_back({im2col, nullptr, nullptr});
      }

    } else if (strategy_code == static_cast<char>(Strategy::TIME_MULTIPLEX)) {
      LOG(FATAL) << "Not Implemented";
    } else {
      LOG(FATAL) << "Strategy Unknown";
    }
  }
}

// shared_ptr<vector<LogicalOperator>> Fijit::generate_query(string model_name,
//                                                           int replica_id) {
//   shared_ptr<vector<LogicalOperator>> ops =
//       model_manager->instantiate_model(model_name, replica_id);

//   auto all_blocks = model_manager->get_all_blocks_config(model_name);
//   for (int i = 0; i < ops->size(); ++i) {
//     ops->at(i).preload(all_blocks, &handle, &cublasHandle);
//   }

//   model_manager->register_input(model_name, replica_id, input,
//                                 input_tensor_name);

//   return ops;
// }

// void Fijit::prepare() {
//   fijit_prepared = true;

//   // So far we support one model one replica and one input
//   for (auto &kv : model_protos) {
//     string model_name = kv.first;
//     LOG(INFO) << fmt::format("Preparing input for model {}", model_name);

//     auto scheduler_queue =
//         make_shared<ReaderWriterQueue<vector<LogicalOperator>>>();
//     auto dispatch_queue =
//         scheduler->register_model_queue(model_name, scheduler_queue);

//     sched_queues.insert({model_name, scheduler_queue});
//     exec_queues.insert({model_name, dispatch_queue});
//     executor->register_queue(model_name, dispatch_queue);

//     shared_ptr<vector<LogicalOperator>> queries = generate_query(model_name);
//     model_to_queries.insert({model_name, queries});
//   }

//   scheduler_thread = thread([&]() { scheduler->start(); });
//   executor_thread = thread([&]() { executor->start(); });
// }

void Fijit::infer() {
  event_registra.record(EventType::BEGIN, EventSource::Executor, "Total", 0);
  for (auto &vliw : op_queue) {
    for (auto op : vliw) {
      if (op == nullptr) {
        continue;
      }
      op->dispatch(default_stream);
    }
  }
  event_registra.record(EventType::END, EventSource::Executor, "Total", 0,
                        default_stream);
  // CHECK(fijit_prepared);

  // auto sleep_time_ns =
  //     chrono::duration_cast<chrono::nanoseconds>(chrono::seconds(1)) / qps;

  // for (size_t i = 0; i < total_query; i++) {
  //   auto start = chrono::high_resolution_clock::now();

  //   for (auto &kv : sched_queues) {
  //     auto queries = model_to_queries.at(kv.first);
  //     kv.second->enqueue(*queries);
  //   }

  //   auto end = chrono::high_resolution_clock::now();
  //   auto processing_time = end - start;
  //   auto time_left_to_sleep = sleep_time_ns - processing_time;

  // LOG(INFO) << fmt::format(
  //     "Sleeping for {}us",
  //     chrono::duration_cast<chrono::microseconds>(time_left_to_sleep)
  //         .count()
  //         );

  // CHECK(time_left_to_sleep > chrono::nanoseconds(1));
  // if (time_left_to_sleep > chrono::nanoseconds(1)) {
  //   std::this_thread::sleep_for(time_left_to_sleep);
  // }
  // }

  // std::this_thread::sleep_for(0.5s);
  // wait_for_queues();

  CHECK_CUDA(cudaDeviceSynchronize());
}

// TODO(simon): Templatize this!
// void wait_for_queue(SchedQueue q) {
//   for (size_t i = 0; i < 2; i++) {
//     while (q->size_approx() != 0) {
//       std::this_thread::sleep_for(10ms);
//     }
//   }
// }

// void wait_for_queue(ExecQueue q) {
//   for (size_t i = 0; i < 2; i++) {
//     while (q->size_approx() != 0) {
//       // LOG(INFO) << fmt::format("ExecQueue size {}", q->size_approx());
//       std::this_thread::sleep_for(10ms);
//     }
//   }
// }

// void Fijit::wait_for_queues() {
//   for (auto &kv : sched_queues) {
//     LOG(INFO) << fmt::format("Waiting for sched queue {} to flush",
//     kv.first); wait_for_queue(kv.second);
//   }
//   for (auto &kv : exec_queues) {
//     LOG(INFO) << fmt::format("Waiting for exec queue {} to flush", kv.first);
//     wait_for_queue(kv.second);
//   }
// }

Fijit::Fijit() {
  // Initialize CUDA Context
  LOG(INFO) << "initializing cuda context";
  cudaCtx = cuda_init();
  cudnnCreate(&handle);
  cublasCreate(&cublasHandle);
  cudaStreamCreate(&default_stream);
}

// Fijit::~Fijit() {
//   executor->stop();
//   scheduler->stop();
//   scheduler_thread.join();
//   executor_thread.join();
// }