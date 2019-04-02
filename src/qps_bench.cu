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
    int max_block_;
    string input_name_;
    int num_query_;
    int num_stream_;
    const string model_name_ = "default-model";
    map<int, shared_ptr<vector<shared_ptr<LogicalOperator>>>> ctx_map_;

    QPSBench(const char *model_path, string input_path, int max_block,
             string input_name, int num_query, int num_stream) {
        max_block_ = max_block;
        input_name_ = input_name;
        num_query_ = num_query;
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
        scheduler_queue_ = make_shared<ConcurrentQueue<shared_ptr<LogicalOperator>>>();
        scheduler_ = make_shared<StaticScheduler>(max_block, &cudaCtx_, &handle_, &cublasHandle_);
        dispatch_queue_ = scheduler_->register_model_queue(model_name_, scheduler_queue_);
        scheduler_thread_ = thread([&]() { scheduler_->start(); });
    }

    QueryContext dispatch_query(int query_id) {
        printf(" \t\t-> Computing query %d\n", query_id);
        const int stream_id = query_id % num_stream_;
        if (ctx_map_.find(stream_id) == ctx_map_.end()) {
            printf(" \t\t\t-> Caching stream %d for query %d\n", stream_id, query_id);
            auto ops = model_manager_->instantiate_model(model_name_, stream_id);
            ctx_map_.insert(make_pair(stream_id, ops));
        }
        auto ops = ctx_map_.at(stream_id);
        model_manager_->register_input(model_name_, query_id, input_, input_name_);

        printf(" \t\t\t-> Enqueuing %lu operations\n", ops->size());
        for (auto o : *ops) {
            while (!scheduler_queue_->enqueue(o)) {
                cerr << "enqueued failed" << endl;
            }
        }

        auto physical_ops = make_shared<vector<shared_ptr<PhysicalOperator>>>();
        // cerr << "Dispatching size " << dispatch_queue_->size_approx() << endl;
        int profile_num_waits = 0;
        int profile_num_ops = 0;
        for (int i = 0; i < ops->size(); ++i) {
            shared_ptr<PhysicalOperator> op;
            while (!dispatch_queue_->try_dequeue(op)) {
                //printf(" \t\t\t-> WAIT, processed %d ops\n", profile_num_ops);
                // this_thread::sleep_for(std::chrono::milliseconds(500));
                // cerr << "\tcan't deque operators, current physical ops size "
                //     << physical_ops->size() << " but waiting now has dispatching size "
                //     << dispatch_queue_->size_approx() << endl;
                profile_num_waits++;
            }
            physical_ops->push_back(op);
            profile_num_ops++;
        }
        printf(" \t\t\t-> Dequeued all physical operations with %d waits\n", profile_num_waits);

        auto events_collection = make_shared<vector<vector<cudaEvent_t>>>(0);
        QueryContext ctx{.logical_ops = ops,
                .physical_ops = physical_ops,
                .events_collection = events_collection};

        return ctx;
    }

    ~QPSBench() {
        scheduler_->stop();
        scheduler_thread_.join();
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

    shared_ptr<ConcurrentQueue<shared_ptr<LogicalOperator>>> scheduler_queue_;
    shared_ptr<ConcurrentQueue<shared_ptr<PhysicalOperator>>> dispatch_queue_;
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
    const string input_path = result["input"].as<string>();
    const int max_block = result["max-block"].as<int>();
    const string input_name = result["input-name"].as<string>();
    const int num_query = result["num-query"].as<int>();
    const int num_stream = result["num-stream"].as<int>();

    QPSBench bench(model_path, input_path, max_block, input_name, num_query, num_stream);
    map<tuple<int, int>, QueryContext> dispatches;

    printf(" -> Preloading kernels for constant resource allocations\n");
    for (int i = 0; i < num_stream; ++i) {
        for (int j = 0; j < num_query; ++j) {
            QueryContext ctx = bench.dispatch_query(i);
            tuple<int, int> id = make_tuple(j, i);
            dispatches.insert({id, ctx});
        }
        printf(" \t-> Stream %d / %d preloading complete\n", i, num_stream);
    }

    printf(" -> Creating streams\n");
    vector<cudaStream_t> streams;
    for (int k = 0; k < num_stream; ++k) {
        cudaStream_t s;
        CHECK_CUDA(cudaStreamCreate(&s));
        streams.push_back(s);
    }

    printf(" -> Dispatching queries\n");
    cudaStream_t timing_stream;
    CHECK_CUDA(cudaStreamCreate(&timing_stream));
    for (int l = 0; l < num_query; ++l) {
        for (int i = 0; i < num_stream; ++i) {
            QueryContext ctx = dispatches.at(make_tuple(l, i));
            cudaStream_t s = streams[i];

            cudaEvent_t dispatch_time;
            CHECK_CUDA(cudaEventCreate(&dispatch_time));
            CHECK_CUDA(cudaEventRecord(dispatch_time, timing_stream));
            vector<cudaEvent_t> time_tup{dispatch_time, dispatch_time};
            ctx.events_collection->emplace_back(time_tup);
            CHECK_CUDA(cudaEventSynchronize(dispatch_time));

            for (int k = 0; k < ctx.physical_ops->size(); ++k) {
                ctx.events_collection->emplace_back(ctx.physical_ops->at(k)->dispatch(s));
            }
        }
    }

    printf(" -> Waiting for query completion\n");
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int q = 0; q < num_query; ++q) {
        for (int m = 0; m < num_stream; ++m) {
            QueryContext ctx = dispatches.at(make_tuple(q, m));
            cudaEvent_t dispatch_time = ctx.events_collection->front()[0];
            cudaEvent_t start_time = ctx.events_collection->at(1)[0];
            cudaEvent_t complete_time = ctx.events_collection->back()[1];

            float dispatch_time_ms, run_time_ms;
            cudaEventElapsedTime(&dispatch_time_ms, dispatch_time, complete_time);
            cudaEventElapsedTime(&run_time_ms, start_time, complete_time);

            printf("stream_id[%d] query_id[%d] dispatch_time=%.4f run_time=%.4f\n", m, q, dispatch_time_ms, run_time_ms);
        }
    }
}
