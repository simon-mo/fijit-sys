#include <cassert>
#include <memory>
#include <vector>

#include "redox.hpp"

#include "common.h"
#include "kernel_db.h"

using namespace redox;
using namespace std;

#define CHECK_OK(c)                                                            \
  if (!c.ok()) {                                                               \
    cerr << "Failed to execute redis, status: " << c.status() << endl;         \
    cerr << __FILE__ << ":" << __LINE__ << endl;                               \
    exit(1);                                                                   \
  }

KernelDB::KernelDB() { rdx.connect("localhost", 6379); }

KernelDB::~KernelDB() { rdx.disconnect(); }

vector<string> split(string s, string delimiter) {
  vector<string> list;
  size_t pos = 0;
  string token;
  while ((pos = s.find(delimiter)) != string::npos) {
    token = s.substr(0, pos);
    list.push_back(token);
    s.erase(0, pos + delimiter.length());
  }
  list.push_back(s);
  return list;
}

bool KernelDB::exists(string &op) {
  auto &c = rdx.commandSync<int>({"EXISTS", op});
  CHECK_OK(c)
  return (bool)c.reply();
}

string KernelDB::get_fatbin(string &op) {

  auto &c = rdx.commandSync<string>({"HGET", op, "fatbin"});
  CHECK_OK(c);

  return c.reply();
}

k_dim3 KernelDB::get_block_dim(string &op) {

  auto &c = rdx.commandSync<string>({"HGET", op, "block_dim"});
  CHECK_OK(c);

  vector<string> dims = split(c.reply(), ",");
  int d1 = atoi(dims[0].c_str());
  int d2 = atoi(dims[1].c_str());
  int d3 = atoi(dims[2].c_str());

  return make_tuple(d1, d2, d3);
}

k_dim3 KernelDB::get_grid_dim(string &op) {

  auto &c = rdx.commandSync<string>({"HGET", op, "grid_dim"});
  CHECK_OK(c);

  vector<string> dims = split(c.reply(), ",");
  int d1 = atoi(dims[0].c_str());
  int d2 = atoi(dims[1].c_str());
  int d3 = atoi(dims[2].c_str());

  return make_tuple(d1, d2, d3);
}

vector<KERNEL_ARG> KernelDB::get_kernel_args(string &op) {
  auto &c = rdx.commandSync<string>({"HGET", op, "kernel_args"});
  CHECK_OK(c);

  vector<string> args = split(c.reply(), ",");
  vector<KERNEL_ARG> k_args;
  for (auto &arg : args) {
    k_args.push_back([](auto &arg) {
      if (arg == "INPUT")
        return INPUT;
      if (arg == "OUTPUT")
        return OUTPUT;
      if (arg == "DATA")
        return DATA;
    }(arg));
  }
  return k_args;
}
