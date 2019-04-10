#include "cuda.h"
#include "events.h"

#include <iostream>
#include <string>

using namespace std;

void EventRegistrar::insert(string model_name, cudaEvent_t start,
                            cudaEvent_t end) {
  EventEntry v = {.model_name = model_name, start, end};
  data.push_back(v);
}

void EventRegistrar::insert(string model_name, vector<cudaEvent_t> events) {
  insert(model_name, events[0], events[1]);
}

vector<vector<cudaEvent_t>> EventRegistrar::get_events(string model_name) {
  vector<vector<cudaEvent_t>> result;
  for (EventEntry &e : data) {
    if (e.model_name != model_name) {
      continue;
    }
    vector<cudaEvent_t> events = {e.begin, e.end};
    result.push_back(events);
  }
  return result;
}

EventRegistrar &EventRegistrar::get_global_event_registrar() {
  static EventRegistrar global_event_registrar;
  return global_event_registrar;
}
