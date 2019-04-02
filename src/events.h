//
// Created by Simon Mo on 2019-04-01.
//

#ifndef FIJIT_SYS_EVENTS_H
#define FIJIT_SYS_EVENTS_H

#include "cuda.h"
#include <list>
#include <memory>
#include <tuple>
#include <vector>

using namespace std;

struct EventEntry {
  string model_name;
  cudaEvent_t begin;
  cudaEvent_t end;
};

class EventRegistrar {
public:
  void insert(string model_name, cudaEvent_t start, cudaEvent_t end);

  void insert(string model_name, vector<cudaEvent_t> events);

  vector<vector<cudaEvent_t>> get_events(string model_name);

  static EventRegistrar &get_global_event_registrar(void);

private:
  list<EventEntry> data;
};

#endif // FIJIT_SYS_EVENTS_H
