#ifndef KERNEL_DB_H
#define KERNEL_DB_H

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common.h"
#include "redox.hpp"

using namespace std;

class KernelDB {
public:
  KernelDB();

  ~KernelDB();

  bool exists(string &);

  string get_fatbin(string &);

  k_dim3 get_block_dim(string &);

  k_dim3 get_grid_dim(string &);

  vector<KERNEL_ARG> get_kernel_args(string &);

  string get_kernel_name(string &);

  static KernelDB &get_global_kernel_db(void);

private:
  redox::Redox rdx;
};

#endif /* KERNEL_DB_H */
