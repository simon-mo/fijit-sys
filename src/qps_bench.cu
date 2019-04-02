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


class QPSBench {
public:
    string input_name_;
    int num_stream_;
    const string model_name_ = "default-model";
    map<int, shared_ptr<vector<shared_ptr<LogicalOperator>>>> ctx_map_;
    map<int, shared_ptr<vector<shared_ptr<PhysicalOperator>>>> ctx_physical_map_;

    QPSBench(const char *model_path, string input_path, int max_block,
             string input_name, int num_stream) {
        input_name_ = input_name;
        num_stream_ = num_stream;

        // CUDA init
        cudaCtx_ = cuda_init();
        cudnnCreate(&handle_);
        cublasCreate(&cublasHandle_);

        // load model
        parse_model(model_, model_path);
        parse_input(input_, input_path);
        auto smm = make_shared<StaticMemoryManager>();
        auto dmm = make_shared<DynamicMemoryManager>();
        model_manager_ = make_shared<ModelManager>(smm, dmm);
        model_manager_->register_model(model_, model_name_);

        // init scheduler
        scheduler_queue_ = make_shared<BlockingConcurrentQueue<shared_ptr<LogicalOperator>>>();
        scheduler_ = make_shared<StaticScheduler>(max_block, &cudaCtx_, &handle_, &cublasHandle_);
        dispatch_queue_ = scheduler_->register_model_queue(model_name_, scheduler_queue_);
        scheduler_thread_ = thread([&]() { scheduler_->start(); });
    }

    QueryContext dispatch_query(int query_id) {
        const int stream_id = query_id % num_stream_;
        if (ctx_map_.find(stream_id) == ctx_map_.end()) {
            auto ops = model_manager_->instantiate_model(model_name_, stream_id);
            model_manager_->register_input(model_name_, stream_id, input_, input_name_);
            ctx_map_.insert(make_pair(stream_id, ops));

            for (auto o : *ops) {
                scheduler_queue_->enqueue(o);
            }

            auto physical_ops = make_shared<vector<shared_ptr<PhysicalOperator>>>();
            for (int i = 0; i < ops->size(); ++i) {
                shared_ptr<PhysicalOperator> op;
                dispatch_queue_->wait_dequeue(op);
                physical_ops->push_back(op);
            }
            ctx_physical_map_.insert(make_pair(stream_id, physical_ops));
        }

        QueryContext ctx{
                .logical_ops = ctx_map_.at(stream_id),
                .physical_ops = ctx_physical_map_.at(stream_id),
                .events_collection = make_shared<vector<vector<cudaEvent_t>>>(0)
        };

        return ctx;
    }

    ~QPSBench() {
        scheduler_->stop();
        scheduler_thread_.join();
        cudnnDestroy(handle_);
        cublasDestroy(cublasHandle_);
        cuCtxDestroy(cudaCtx_);
    }


private:
    CUcontext cudaCtx_;
    cudnnHandle_t handle_;
    cublasHandle_t cublasHandle_;
    ModelProto model_;
    TensorProto input_;

    shared_ptr<ModelManager> model_manager_;
    shared_ptr<StaticScheduler> scheduler_;
    thread scheduler_thread_;

    shared_ptr<BlockingConcurrentQueue<shared_ptr<LogicalOperator>>> scheduler_queue_;
    shared_ptr<BlockingConcurrentQueue<shared_ptr<PhysicalOperator>>> dispatch_queue_;
};


int main(int argc, char *argv[]) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    cxxopts::Options options("qps_bench", "FIJIT Inference Engine benchmark (QPS)");
    options.positional_help("[optional args]").show_positional_help();
    // clang-format off
    options.add_options()
            ("m,model", "Path to the model ONNX file", cxxopts::value<string>())
            ("i,input", "Path to the input ONNX file", cxxopts::value<string>())
            ("max-block", "Max block for TVM ops", cxxopts::value<int>()->default_value("80"))
            ("input-name", "Override input tensor name", cxxopts::value<string>()->default_value("0"))
            ("num-stream", "Number of stream", cxxopts::value<int>()->default_value("1"))

            ("num-query-burst", "Number of query per stream (burst)", cxxopts::value<int>()->default_value("1"))
            ("num-bursts", "Number of bursts", cxxopts::value<int>()->default_value("10"))
            ("inter-burst-us", "Time between bursts in usec", cxxopts::value<int>()->default_value("100"))

            ("h, help", "Print help");
    // clang-format on
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(0);
    }

    const char *model_path = result["model"].as<string>().c_str();
    const string input_path = result["input"].as<string>();
    const int max_block = result["max-block"].as<int>();
    const string input_name = result["input-name"].as<string>();
    const int num_stream = result["num-stream"].as<int>();

    const int num_query_burst = result["num-query-burst"].as<int>();
    const int num_bursts = result["num-bursts"].as<int>();
    const int inter_burst_us = result["inter-burst-us"].as<int>();
    const int num_query = num_query_burst * num_bursts;  // num queries per stream

    QPSBench bench(model_path, input_path, max_block, input_name, num_stream);
    map<int, QueryContext> dispatches;

    // printf(" -> Preloading kernels for constant resource allocations\n");
    for (int j = 0; j < num_query; ++j) {
        QueryContext ctx = bench.dispatch_query(j);
        dispatches.insert({j, ctx});
    }

    // printf(" -> Creating streams\n");
    vector<cudaStream_t> streams;
    for (int k = 0; k < num_stream; ++k) {
        cudaStream_t s;
        CHECK_CUDA(cudaStreamCreate(&s));
        streams.push_back(s);
    }

    // printf(" -> Dispatching queries\n");
    cudaStream_t timing_stream;
    CHECK_CUDA(cudaStreamCreate(&timing_stream));
    for (int l = 0; l < num_query; ++l) {
        QueryContext ctx = dispatches.at(l);
        cudaStream_t s = streams[l % num_stream];

        cudaEvent_t dispatch_time;
        CHECK_CUDA(cudaEventCreate(&dispatch_time));
        CHECK_CUDA(cudaEventRecord(dispatch_time, timing_stream));

        vector<cudaEvent_t> time_tup{dispatch_time, dispatch_time};
        ctx.events_collection->emplace_back(time_tup);

        // TODO create cuda events
        //    cudaEventCreate(&start_event);
        //    cudaEventCreate(&end_event);
        for (int k = 0; k < ctx.physical_ops->size(); ++k) {
            ctx.events_collection->emplace_back(ctx.physical_ops->at(k)->dispatch(s));
        }

        // CHECK_CUDA(cudaEventSynchronize(dispatch_time));

        if ((l + 1) % num_query_burst == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(inter_burst_us));
        }
    }

    // printf(" -> Waiting for query completion\n");
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int q = 0; q < num_query; ++q) {
        QueryContext ctx = dispatches.at(q);
        cudaEvent_t dispatch_time = ctx.events_collection->front()[0];
        cudaEvent_t start_time = ctx.events_collection->at(1)[0];
        cudaEvent_t complete_time = ctx.events_collection->back()[1];

        float dispatch_time_ms, run_time_ms;
        cudaEventElapsedTime(&dispatch_time_ms, dispatch_time, complete_time);
        cudaEventElapsedTime(&run_time_ms, start_time, complete_time);

        // for (auto &event_list : ctx.events_collection.get()) {
        //     for (auto &event : event_list) {
        //         cudaEventDestroy(event);
        //     }
        // }

        printf("{\"stream_id\": %d, \"query_id\": %d, \"dispatch_time\": %.4f, \"run_time\": %.4f, \"total_time\": %.4f}\n",
                q % num_stream, q, dispatch_time_ms, run_time_ms, dispatch_time_ms + run_time_ms);
    }

    // for (auto &stream : streams) {
    //     cudaStreamDestroy(stream);
    // }
}
