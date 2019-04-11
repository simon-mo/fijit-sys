#include "reporter.h"

#include "fmt/core.h"

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "events.h"

#include <chrono>

using namespace std;
using namespace rapidjson;
using namespace chrono;

string report_chrome_trace(void) {
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

  vector<EventEntry> entries =
      EventRegistrar::get_global_event_registrar().get_events();

  Document document;
  Document::AllocatorType &allocator = document.GetAllocator();

  document.SetArray();

  for (EventEntry &e : entries) {
    Value name;
    name.SetString(e.name.c_str(), e.name.size(), allocator);

    Value event_type;
    string event_str(1, static_cast<char>(e.type));
    event_type.SetString(event_str.c_str(), event_str.size(), allocator);

    auto timestamp = duration_cast<microseconds>(nanoseconds(e.ts_ns));

    Value item(kObjectType);
    item.AddMember("pid", Value(1), allocator);
    item.AddMember("tid", Value(e.tid), allocator);
    item.AddMember("ts", timestamp.count(), allocator);
    item.AddMember("ph", event_type, allocator);
    item.AddMember("name", name, allocator);
    document.PushBack(item.Move(), allocator);
  }

  StringBuffer buffer;
  Writer<StringBuffer> writer(buffer);
  document.Accept(writer);

  return buffer.GetString();
}
