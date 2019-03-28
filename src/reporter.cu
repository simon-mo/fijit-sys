#include <sstream>


#include "cuda.h"
#include "fmt/core.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "reporter.h"

using namespace std;
using namespace rapidjson;

string ChromeTraceReporter::report() {
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

  return buffer.GetString();
}

string TotalTimeReporter::report() {
  float total_time;
  cudaEventElapsedTime(&total_time, start_of_the_world,
                       events_collection->at(events_collection->size() - 1)[1]);
  return fmt::format("Total time: {} ms", total_time);
}

string ModelTimeReporter::report() {
  const int nops = physical_ops->size();
  const int ntrials = events_collection->size() / nops;

  std::stringstream output;
  for (int trialid = 0; trialid < ntrials; trialid++) {
    float total_time = 0.0;
    for (int opid = 0; opid < nops; opid++) {
      float op_time;
      vector<cudaEvent_t> times = events_collection->at(trialid * nops + opid);
      cudaEventElapsedTime(&op_time, times[0], times[1]);
      total_time += op_time;
    }
    output << total_time << "\t";
  }

  return output.str();
}
