#include "events.h"

#include "common/common_cuda.h"

#include <chrono>
#include <iostream>
#include <string>

#include "glog/logging.h"

using namespace std;

void EventRegistrar::record(EventType type, EventSource source, string name,
                            int tid) {
  int64_t now =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  EventEntry e{type, source, name, now, tid};
  auto entry = make_unique<EventEntry>(e);
  data.push_back(move(entry));
}

void EventRegistrar::record(EventType type, EventSource source, string name,
                            int tid, cudaStream_t s) {
  EventEntry e{type, source, name, 0, tid};
  auto entry = make_unique<EventEntry>(e);
  CHECK_CUDA(cudaStreamAddCallback(s, host_record_time, &entry->ts_ns, 0));
  data.push_back(move(entry));
}

vector<EventEntry> EventRegistrar::get_events() {
  vector<EventEntry> entries;
  for (auto &e : data) {
    entries.push_back(*e);
  }
  return entries;
}

vector<EventEntry> EventRegistrar::get_events(string name) {
  vector<EventEntry> entries;
  for (auto &e : data) {
    if (e->name == name) {
      entries.push_back(*e);
    }
  }
  return entries;
}

EventRegistrar &EventRegistrar::get_global_event_registrar() {
  static EventRegistrar global_event_registrar;
  return global_event_registrar;
}

std::ostream &operator<<(std::ostream &os, EventType t) {
  return os << static_cast<char>(t);
}