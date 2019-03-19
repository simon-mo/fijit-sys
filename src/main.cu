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


using namespace rapidjson;

using namespace std;
using namespace onnx;
using namespace google::protobuf::io;

int main(int argc, char *argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  assert(argc == 4);

  cuda_init();
  cudnnHandle_t handle;
  cudnnCreate(&handle);
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);

  ModelProto model;
  {
    int fd = open(argv[1], O_RDONLY);
    ZeroCopyInputStream *raw_input = new FileInputStream(fd);
    CodedInputStream *coded_input = new CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(671088640, 167108860);

    model.ParseFromCodedStream(coded_input);

    close(fd);
  }

  TensorProto input;
  {
    fstream f(argv[2], ios::in | ios::binary);
    input.ParseFromIstream(&f);
  }

  string input_name = argv[3];

  // Step 1, malloc all data
  shared_ptr<MemoryManager> mm = make_shared<MemoryManager>();

  shared_ptr<unordered_map<string, ValueInfoProto>> shape_map =
      traverse_shapes(model.graph());

  {
    for (auto name_to_val : *shape_map) {
      mm->register_placeholder(name_to_val.second);
    }
  }

  // Step 2, fill in the placeholder
  for (auto tensor : model.graph().initializer()) {
    mm->register_tensor(tensor);
  }

  mm->register_tensor(input, input_name);

  // Step 3, realize the operators
  vector<LogicalOperator> ops;
  for (auto n : model.graph().node()) {
    ops.emplace_back(n, shape_map);
  }

  int max_block = 20;
  vector<unique_ptr<PhysicalOperator>> physical_ops;

  for (auto o : ops) {
    physical_ops.push_back(o.realize(max_block, &handle, &cublasHandle));
  }

  // Step 4, connect the graph
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
      cudaEventSynchronize(
          events_collection[events_collection.size() - 1][1]
          ));
  CHECK_CUDA(cudaDeviceSynchronize());


//  for (auto out : model.graph().output()) {
//    float *result = mm->get_value(out.name());
//    for (int i = 0; i < 4; ++i) {
//      cout << result[i] << " ";
//    }
//    cout << endl;
//  }

//  [
//  { "pid":1, "tid":1, "ts":87705, "dur":956189, "ph":"X", "name":"Jambase", "args":{ "ms":956.2 } },
//  { "pid":1, "tid":1, "ts":128154, "dur":75867, "ph":"X", "name":"SyncTargets", "args":{ "ms":75.9 } },
//  { "pid":1, "tid":1, "ts":546867, "dur":121564, "ph":"X", "name":"DoThings", "args":{ "ms":121.6 } }
//  ]

  Document document;
  Document::AllocatorType& allocator = document.GetAllocator();

  document.SetArray();

  for (int k = 0; k < physical_ops.size(); ++k) {


    float duration, start_time;
    cudaEvent_t start = events_collection[k][0];
    cudaEvent_t end = events_collection[k][1];

    cudaEventElapsedTime(&duration, start, end);
    cudaEventElapsedTime(&start_time, start_of_world, start);

    string op_name = physical_ops[k]->get_name();
    string formatted_op_name = fmt::format("{}-{:x}", op_name, k);


    Value name;
    name.SetString(formatted_op_name.c_str(), formatted_op_name.size(), allocator);

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

// Output {"project":"rapidjson","stars":11}
  std::cout << buffer.GetString() << std::endl;
}
