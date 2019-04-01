//
// Created by Simon Mo on 2019-04-01.
//

#ifndef FIJIT_SYS_EVENTS_H
#define FIJIT_SYS_EVENTS_H

#include "cuda.h"
#include <list>
#include <tuple>
#include <vector>

using namespace std;

struct EventEntry {
  string model_name;
  int queue_id;
  cudaEvent_t begin;
  cudaEvent_t end;
};

class EventRegistrar {
public:
  void insert(string model_name, int queue_id, cudaEvent_t start,
              cudaEvent_t end) {
    EventEntry v = {.model_name = model_name, .queue_id = queue_id, start, end};
    data.push_back(v);
  }

  void insert(string model_name, int queue_id, vector<cudaEvent_t> events) {
    insert(model_name, queue_id, events[0], events[1]);
  }

private:
  list<EventEntry> data;
};

shared_ptr<EventRegistrar> global_event_registrar = nullptr;

shared_ptr<EventRegistrar> get_gobal_event_registrar() {
  if (global_event_registrar == nullptr) {
    global_event_registrar = make_shared<EventRegistrar>();
  }
  return global_event_registrar;
}

#endif // FIJIT_SYS_EVENTS_H
