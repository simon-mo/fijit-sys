#ifndef RUNTIME_COMMON_H
#define RUNTIME_COMMON_H

#include <chrono>
#include <map>
#include <list>

#include "readerwriterqueue/readerwriterqueue.h"
#include "proto/onnx.pb.h"

#include "operators/operators.h"

using namespace std;
using namespace onnx;
using namespace moodycamel;

struct Request {
  string model_name;
  // shared_ptr<TensorProto> inp;
  chrono::time_point<chrono::microseconds> deadline;
  int query_id;
};

// e.g. {
//     "res50": [Req1 {"res50", &inp_tensor, deadline_a}, Req2 ...],
//     "inception": [Req 3 {"inception", &inp_tensor, deadline_c}]
// }
using RequestBatch = multimap<string, Request>;
using RequestQueue = shared_ptr<BlockingReaderWriterQueue<RequestBatch>>;

// e.g. [Conv, Conv, Conv], [BN, Relu]
using VLIW = list<LogicalOperator>;
using InstQueue = shared_ptr<BlockingReaderWriterQueue<VLIW>>;

#endif /* RUNTIME_COMMON_H */
