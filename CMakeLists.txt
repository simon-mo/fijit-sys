cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(fijit-sys LANGUAGES CXX CUDA)
enable_language(CXX)

set(CMAKE_CXX_COMPILER)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

include(cmake/cuda.cmake)
include(cmake/FindLibunwind.cmake)

add_subdirectory(proto)
add_subdirectory(third_party/redox EXCLUDE_FROM_ALL)
add_subdirectory(third_party/rapidjson)
add_subdirectory(third_party/glog EXCLUDE_FROM_ALL)
add_subdirectory(third_party/fmt)

macro(CUDA_TARGET)
    target_compile_features(${ARGV} PUBLIC cxx_std_14)
    set_target_properties(${ARGV} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endmacro(CUDA_TARGET)

# Our Main Executable is fijit-sys binary
add_executable(${PROJECT_NAME}
               src/main.cu
               src/fijit.h
               src/fijit.cu
               third_party/cxxopts/cxxopts.hpp)
CUDA_TARGET(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
                      proto
                      ${PROTOBUF_LIBRARY}
                      ${LIBUNWIND_LIBRARIES}
                      glog::glog)

add_library(fijit-utils
    src/utils/backtrace.h
    src/utils/onnx_helper.h
    src/utils/onnx_helper.cc
    src/utils/kernel_db.cc
    src/utils/kernel_db.h
    src/common/common.h
)
target_link_libraries(fijit-utils redox proto)
target_compile_features(fijit-utils PUBLIC cxx_std_14)

# GPU Only Libraries
add_library(common-cuda
    src/common/common_cuda.h
    src/common/common_cuda.cu
)
target_link_libraries(common-cuda
    CUDA::cudart
    CUDA::cuda
    CUDA::cudnn
    CUDA::cublas
)
CUDA_TARGET(common-cuda)

add_library(operators
    src/operators/abstract_operators.h
    src/operators/cublas_wrapper.cu
    src/operators/cublas_wrapper.h
    src/operators/cudnn_wrapper.cu
    src/operators/cudnn_wrapper.h
    src/operators/operators.cu
    src/operators/operators.h
)
target_link_libraries(operators
    common-cuda
    fijit-utils
    fmt::fmt
)
CUDA_TARGET(operators)

add_library(runtime
    src/runtime/events.cu
    src/runtime/events.h
    src/runtime/executor.cu
    src/runtime/executor.h
    src/runtime/memory_manager.cu
    src/runtime/memory_manager.h
    src/runtime/model_manager.cu
    src/runtime/model_manager.h
    src/runtime/reporter.cu
    src/runtime/reporter.h
    src/runtime/scheduler.cu
    src/runtime/scheduler.h
)
target_link_libraries(runtime
    common-cuda
    operators
    fijit-utils
    rapidjson
    fmt::fmt
)
CUDA_TARGET(runtime)
target_compile_definitions(runtime
   PUBLIC RAPIDJSON_WRITE_DEFAULT_FLAGS=2
)
                      
target_link_libraries(${PROJECT_NAME} fijit-utils common-cuda operators runtime)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(GoogleBenchmark REQUIRED)
include_directories(${benchmark_INCLUDE_DIRS})
add_executable(benchmark
    src/benchmarks/ops.cu
)
CUDA_TARGET(benchmark)
target_link_libraries(benchmark ${benchmark_LIBRARIES} pthread fijit-utils common-cuda operators runtime 
                        proto
                      ${PROTOBUF_LIBRARY}
                      glog::glog
)



