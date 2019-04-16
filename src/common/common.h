#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <tuple>
#include <unordered_map>

using namespace std;

enum KERNEL_ARG { INPUT, OUTPUT, DATA };
typedef tuple<int, int, int> k_dim3;

#endif /* COMMON_H */
