#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <iostream>
#include <limits>
#include <memory.h>
#include <tuple>
#include <vector>

#include "common.h"
#include "common_cuda.h"
#include "cublas_v2.h"
#include "cuda_runtime_api.h"
#include "cudnn.h"

#include "cublas_wrapper.h"
#include "kernel_db.h"
#include "operators.h"

#include <fmt/core.h>
#include <vector>

using namespace std;

unordered_set<string> AllowedOps({"Conv", "AveragePool", "MaxPool", "Add",
                                  "Sum", "Relu", "BatchNormalization",
                                  "Softmax", "Gemm", "Reshape",
                                  "GlobalAveragePool", "Flatten"});

class NoSuchOpException : public exception {
    virtual const char *what() const noexcept {
        return fmt::format("Can't find this operation").c_str();
    }
};

TVMOperator::TVMOperator(string binary, k_dim3 k_block, k_dim3 k_grid,
                         vector<KERNEL_ARG> k_args, string kernel_name) {
    CUmodule mod;
    CHECK_CUDEVICE(cuModuleLoadFatBinary(&mod, binary.c_str()));
    CHECK_CUDEVICE(cuModuleGetFunction(&func, mod, kernel_name.c_str()));

    {
        int x, y, z;
        tie(x, y, z) = k_block;
        block = new dim3(x, y, z);
    }

    {
        int x, y, z;
        tie(x, y, z) = k_grid;
        grid = new dim3(x, y, z);
    }

    args = k_args;
}

void TVMOperator::set_argument(KERNEL_ARG arg, CUdeviceptr ptr) {
    switch (arg) {
        case (INPUT):
            input = ptr;
            input_is_set = true;
            break;
        case (DATA):
            if (data_is_set) {
                throw runtime_error(
                        "TVMOperator: can't set data twice, is there a bias term?");
            }
            data = ptr;
            data_is_set = true;
            break;
        case (OUTPUT):
            output = ptr;
            output_is_set = true;
            break;
        default:;
    }
}

void TVMOperator::dispatch(cudaStream_t s) {
    /*cudaEvent_t start_event, end_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);*/

    void **params = new void *[args.size()];
    for (int i = 0; i < args.size(); i++) {
        KERNEL_ARG arg = args[i];
        switch (arg) {
            case (INPUT):
                assert(input_is_set);
                params[i] = &input;
                break;
            case (DATA):
                assert(data_is_set);
                params[i] = &data;
                break;
            case (OUTPUT):
                assert(output_is_set);
                params[i] = &output;
                break;
            default:;
        }
    }

    // cudaEventRecord(start_event, s);

    CHECK_CUDEVICE(cuLaunchKernel(
            /*CUfunction*/ func,
            /*gridDimX*/ (*block).x,
            /*gridDimY*/ (*block).y,
            /*gridDimZ*/ (*block).z,
            /*blockDimX*/ (*grid).x,
            /*blockDimY*/ (*grid).y,
            /*blockDimZ*/ (*grid).z,
            /*sharedMemBytes*/ 0,
            /*hStream*/ s,
            /*void** kernelParams*/ params,
            /* void** extra*/ NULL));

    // cudaEventRecord(end_event, s);
    // return vector<cudaEvent_t>({start_event, end_event});
}

bool TVMOperator::operator==(const TVMOperator &rhs) const {
    return func == rhs.func && block == rhs.block && grid == rhs.grid &&
           args == rhs.args;
}

LogicalOperator::LogicalOperator(NodeProto node_,
                                 decltype(io_shapes) shape_map) {
    node = node_;

    type = node.op_type();

    assert(AllowedOps.find(type) != AllowedOps.end());

    // Check we have all the shape info
    io_shapes = shape_map;
    for (auto i : node.input()) {
        assert(io_shapes->find(i) != io_shapes->end());
    }
    for (auto o : node.output()) {
        assert(io_shapes->find(o) != io_shapes->end());
    }

    input_shape = io_shapes->at(node.input().Get(0));

    output_shape = io_shapes->at(node.output().Get(0));
}

shared_ptr<TVMOperator> LogicalOperator::realize_tvm(int max_blocks) {
    assert(type == "Conv");
    KernelDB db;

    auto shape_vectors = [](ValueInfoProto p) {
        vector<int> shapes(0);
        for (auto d : p.type().tensor_type().shape().dim()) {
            shapes.push_back(d.dim_value());
        }
        return shapes;
    };

    auto inp_shape_vector = shape_vectors(input_shape);
    auto out_shape_vector = shape_vectors(output_shape);

    auto attribute_vectors = [](NodeProto n, string attribute) {
        vector<int> values;
        for (auto attri : n.attribute()) {
            if (attri.name() == attribute) {
                for (auto val : attri.ints()) {
                    values.push_back(val);
                }
            }
        }
        return values;
    };

    auto kernel_shape = attribute_vectors(node, "kernel_shape");
    auto pads = attribute_vectors(node, "pads");
    auto strides = attribute_vectors(node, "strides");
    auto dilations = attribute_vectors(node, "dilations");

    if (pads.size() == 0) {
        pads.push_back(0);
        pads.push_back(0);
    }

    if (dilations.size() == 0) {
        dilations.push_back(1);
        dilations.push_back(1);
    }

    vector<int> kernel_tensor = {out_shape_vector[1], inp_shape_vector[1],
                                 kernel_shape[0], kernel_shape[1]};

    auto python_tuple_fmt_2 = [](vector<int> v) {
        return fmt::format("({}, {})", v[0], v[1]);
    };

    auto python_tuple_fmt_4 = [](vector<int> v) {
        return fmt::format("({}, {}, {}, {})", v[0], v[1], v[2], v[3]);
    };

    //  f"Conv-{job.input_dims}" \
//            f"-{job.kernel_dims}"\
//            f"-{job.stride}"\
//            f"-{job.padding}"\
//            f"-{job.atrous}"\
//            f"-{job.block_constraint}"

    string redis_key = fmt::format(
            "Conv-{}-{}-{}-{}-{}-{}", python_tuple_fmt_4(inp_shape_vector),
            python_tuple_fmt_4(kernel_tensor), python_tuple_fmt_2(strides),
            python_tuple_fmt_2(pads), python_tuple_fmt_2(dilations), max_blocks);

    if (!db.exists(redis_key)) {
        throw NoSuchOpException();
    }

    auto op = make_shared<TVMOperator>(
            db.get_fatbin(redis_key), db.get_block_dim(redis_key),
            db.get_grid_dim(redis_key), db.get_kernel_args(redis_key),
            db.get_kernel_name(redis_key));

    inject_kwargs(op);

    return op;
}

shared_ptr<CUDNNOperator>
LogicalOperator::realize_cudnn(cudnnHandle_t *handle) {
    shared_ptr<CUDNNOperator> ptr;
    if (type == "AveragePool" || type == "GlobalAveragePool") {
        ptr = make_shared<PoolingOperator>(
                handle, input_shape, node, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
    } else if (type == "MaxPool") {
        ptr = make_shared<PoolingOperator>(handle, input_shape, node,
                                           CUDNN_POOLING_MAX);
    } else if (type == "Sum" || type == "Add") {
        ptr = make_shared<AddOperator>(handle, input_shape, output_shape);
    } else if (type == "Relu") {
        ptr = make_shared<ReluOperator>(handle, input_shape);
    } else if (type == "BatchNormalization") {
        double epsilon = std::numeric_limits<double>::max();
        for (auto attri : node.attribute()) {
            if (attri.name() == "epsilon") {
                epsilon = attri.f();
            }
        }
        if (epsilon == std::numeric_limits<double>::max()) {
            throw runtime_error("Can't find epsilon for bn");
        }
        ptr = make_shared<BatchNormOperator>(handle, input_shape, epsilon);
    } else if (type == "Softmax") {
        ptr = make_shared<SoftMaxOperator>(handle, input_shape);
    } else if (type == "Conv") {
        ptr = make_shared<ConvOperator>(handle, node, io_shapes);
    } else {
        throw runtime_error(
                "LogicalOperator::realize_cudnn only accepts AvgPool MaxPool\n");
    }
    inject_kwargs(ptr);
    return ptr;
}

shared_ptr<CUBLASOperator>
LogicalOperator::realize_cublas(cublasHandle_t *cublasHandle) {
    shared_ptr<CUBLASOperator> ptr =
            make_shared<GemmOperator>(cublasHandle, node, io_shapes);
    inject_kwargs(ptr);
    return ptr;
}

shared_ptr<PhysicalOperator>
LogicalOperator::realize(int max_blocks, cudnnHandle_t *handle,
                         cublasHandle_t *cublasHandle) {
    // TODO(simon): this should be moved to the eventual scheduler

    if (type == "Conv") {
        try {
            return realize_tvm(max_blocks);
        } catch (NoSuchOpException &e) {
            return realize_cudnn(handle);
        }

    } else if (type == "AveragePool" || type == "MaxPool" || type == "Sum" ||
               type == "Add" || type == "Relu" || type == "BatchNormalization" ||
               type == "Softmax" || type == "GlobalAveragePool") {
        return realize_cudnn(handle);
    } else if (type == "Gemm") {
        return realize_cublas(cublasHandle);
    } else if (type == "Reshape" || type == "Flatten") {

        int dim = 1;
        for (auto d : input_shape.type().tensor_type().shape().dim()) {
            dim *= d.dim_value();
        }
        auto ptr = make_shared<ReshapeOperator>(dim);
        inject_kwargs(ptr);
        return ptr;

    } else {
        throw runtime_error("Can't realize logical operator");
    }
}
void ReshapeOperator::set_argument(KERNEL_ARG arg, CUdeviceptr ptr) {
    switch (arg) {
        case (INPUT):
            input = ptr;
            input_is_set = true;
            break;
        case (OUTPUT):
            output = ptr;
            output_is_set = true;
            break;
        default:;
    }
}

void ReshapeOperator::dispatch(cudaStream_t stream) {
    // cudaEvent_t start_event, end_event;
    // cudaEventCreate(&start_event);
    // cudaEventCreate(&end_event);
    assert(input_is_set && output_is_set);
    // cudaEventRecord(start_event, stream);
    cuMemcpyDtoDAsync(output, input, sizeof(float) * total_size, stream);
    // cudaEventRecord(end_event, stream);
    // return {start_event, end_event};
}

ReshapeOperator::ReshapeOperator(int total_size) : total_size(total_size) {

}
