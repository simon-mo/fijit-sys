//
// Created by Simon Mo on 2019-04-01.
//

#ifndef FIJIT_SYS_EVENTS_H
#define FIJIT_SYS_EVENTS_H

#include "common/common.h"
#include "common/common_cuda.h"
#include <chrono>
#include <list>
#include <memory>
#include <tuple>
#include <vector>

#include "fmt/ostream.h"

using namespace std;

struct EventEntry {
  EventType type;
  EventSource source;
  string name;
  int64_t ts_ns;
  int tid;

  friend std::ostream &operator<<(ostream &out, const EventEntry &entry) {
    out << fmt::format(" Event .type={}, .name={}, .ts_ns={} ", entry.type,
                       entry.name, entry.ts_ns);
    return out;
  }
};

class EventRegistrar {
public:
  void record(EventType type, EventSource source, string name, int tid);
  void record(EventType type, EventSource source, string name, int tid,
              cudaStream_t s);

  vector<EventEntry> get_events();
  vector<EventEntry> get_events(string name);

  static EventRegistrar &get_global_event_registrar(void);

private:
  list<unique_ptr<EventEntry>> data;
};

#endif // FIJIT_SYS_EVENTS_H
