#include "cuda.h"
#include "fmt/core.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "reporter.h"

using namespace std;
using namespace rapidjson;

string ChromeTraceReporter::report() { return report(1, 1); }

string ChromeTraceReporter::report(int tid) { return report(1, tid); }

string ChromeTraceReporter::report(int pid, int tid) {
  /*
   *   //  [
  //  { "pid":1, "tid":1, "ts":87705, "dur":956189, "ph":"X", "name":"Jambase",
  //  "args":{ "ms":956.2 } },
  //  { "pid":1, "tid":1, "ts":128154, "dur":75867, "ph":"X",
  //  "name":"SyncTargets", "args":{ "ms":75.9 } },
  //  { "pid":1, "tid":1, "ts":546867, "dur":121564, "ph":"X",
  //  "name":"DoThings", "args":{ "ms":121.6 } }
  //  ]
   */
  Document document;
  Document::AllocatorType &allocator = document.GetAllocator();

  document.SetArray();

  for (int k = 0; k < physical_ops->size(); ++k) {
    float duration, start_time;
    cudaEvent_t start = events_collection->at(k)[0];
    cudaEvent_t end = events_collection->at(k)[1];

    cudaEventElapsedTime(&duration, start, end);
    cudaEventElapsedTime(&start_time, start_of_the_world, start);

    string op_name = physical_ops->at(k)->get_name();
    string formatted_op_name = fmt::format("{}-{}", op_name, k);

    Value name;
    name.SetString(formatted_op_name.c_str(), formatted_op_name.size(),
                   allocator);

    Value item(kObjectType);
    item.AddMember("pid", Value(pid), allocator);
    item.AddMember("tid", Value(tid), allocator);
    item.AddMember("ts", start_time, allocator);
    item.AddMember("dur", duration, allocator);
    item.AddMember("ph", Value("X"), allocator);
    item.AddMember("name", name, allocator);
    document.PushBack(item.Move(), allocator);
  }

  StringBuffer buffer;
  Writer<StringBuffer> writer(buffer);
  document.Accept(writer);

  return buffer.GetString();
}

string TotalTimeReporter::report() {
  float total_time;
  cudaEventElapsedTime(&total_time, start_of_the_world,
                       events_collection->at(events_collection->size() - 1)[1]);
  return fmt::format("Total time: {} ms", total_time);
}