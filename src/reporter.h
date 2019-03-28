//
// Created by Simon Mo on 2019-03-19.
//

#ifndef FIJIT_SYS_REPORTER_H
#define FIJIT_SYS_REPORTER_H

#include "cuda.h"

#include <memory>
#include <vector>

#include "operators.h"

using namespace std;

class AbstractReporter {
public:
  AbstractReporter(
      shared_ptr<vector<LogicalOperator>> logical_ops,
      shared_ptr<vector<shared_ptr<PhysicalOperator>>> physical_ops,
      shared_ptr<vector<vector<cudaEvent_t>>> events_collection,
      cudaEvent_t start_of_the_world)
      : logical_ops(logical_ops), physical_ops(physical_ops),
        events_collection(events_collection),
        start_of_the_world(start_of_the_world){};

  virtual string report() = 0;

  shared_ptr<vector<LogicalOperator>> logical_ops;
  shared_ptr<vector<shared_ptr<PhysicalOperator>>> physical_ops;
  shared_ptr<vector<vector<cudaEvent_t>>> events_collection;
  cudaEvent_t start_of_the_world;
};

class ChromeTraceReporter : public AbstractReporter {
public:
  ChromeTraceReporter(
      shared_ptr<vector<LogicalOperator>> logical_ops,
      shared_ptr<vector<shared_ptr<PhysicalOperator>>> physical_ops,
      shared_ptr<vector<vector<cudaEvent_t>>> events_collection,
      cudaEvent_t start_of_the_world)
      : AbstractReporter(logical_ops, physical_ops, events_collection,
                         start_of_the_world){};

  string report() override;
};

class TotalTimeReporter : public AbstractReporter {
public:
  TotalTimeReporter(
      shared_ptr<vector<LogicalOperator>> logical_ops,
      shared_ptr<vector<shared_ptr<PhysicalOperator>>> physical_ops,
      shared_ptr<vector<vector<cudaEvent_t>>> events_collection,
      cudaEvent_t start_of_the_world)
      : AbstractReporter(logical_ops, physical_ops, events_collection,
                         start_of_the_world){};

  string report() override;
};

class ModelTimeReporter : public AbstractReporter {
public:
    ModelTimeReporter(
            shared_ptr<vector<LogicalOperator>> logical_ops,
            shared_ptr<vector<shared_ptr<PhysicalOperator>>> physical_ops,
    shared_ptr<vector<vector<cudaEvent_t>>> events_collection,
    cudaEvent_t start_of_the_world)
    : AbstractReporter(logical_ops, physical_ops, events_collection,
            start_of_the_world){};

    string report() override;
};

#endif // FIJIT_SYS_REPORTER_H
