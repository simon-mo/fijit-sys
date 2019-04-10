cmake_minimum_required(VERSION 3.8 FATAL_ERROR)


project(fijit-sys LANGUAGES CXX CUDA)
enable_language(CXX)


set(CMAKE_CXX_COMPILER)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

include(cmake/cuda.cmake)
include(cmake/FindLibunwind.cmake)

add_subdirectory(proto)
add_subdirectory(third_party/redox EXCLUDE_FROM_ALL)
add_subdirectory(third_party/rapidjson)
add_subdirectory(third_party/glog EXCLUDE_FROM_ALL)
add_subdirectory(third_party/fmt)


## Our Main Executable is fijit-sys binary
add_executable(
        ${PROJECT_NAME}
        src/main.cu
        src/abstract_operators.h
        third_party/cxxopts/cxxopts.hpp
)

target_link_libraries(${PROJECT_NAME}
        proto
        ${PROTOBUF_LIBRARY}
        ${LIBUNWIND_LIBRARIES}
        glog::glog
)


## CPU Only Libraries
add_library(fijit-lib SHARED
        src/kernel_db.h
        src/kernel_db.cc
        src/common.h
        src/onnx_helper.h
        src/onnx_helper.cc 
        src/backtrace.h
        )
target_link_libraries(fijit-lib redox)
target_link_libraries(${PROJECT_NAME} fijit-lib)
target_compile_features(fijit-lib PUBLIC cxx_std_14)
set_target_properties(fijit-lib PROPERTIES LINKER_LANGUAGE CXX)


## GPU Only Libraries
add_library(
        cuda_common STATIC
        src/common_cuda.cc
        src/common_cuda.h
        )
add_library(cuda_lib STATIC
        src/common.h
        src/operators.h
        src/operators.cu
        src/memory_manager.h
        src/memory_manager.cu
        src/kernel_db.h

        src/cudnn_wrapper.cu
        src/cudnn_wrapper.h

        src/onnx_helper.h
        src/abstract_operators.h
        src/cublas_wrapper.h
        src/cublas_wrapper.cu
        src/reporter.h
        src/reporter.cu
        src/model_manager.h
        src/model_manager.cu

        src/scheduler.cu
        src/scheduler.h

        third_party/concurrentqueue/blockingconcurrentqueue.h
        third_party/concurrentqueue/concurrentqueue.h
        src/executor.h
        src/executor.cu
        src/events.h
        src/events.cu
        src/fijit.h
        src/fijit.cu
        src/onnx_helper.h
        )

target_link_libraries(cuda_lib
        cuda_common
        redox
        CUDA::cudart CUDA::cuda CUDA::cudnn CUDA::cublas
        fijit-lib
        fmt::fmt
        rapidjson
        proto
        )
target_link_libraries(${PROJECT_NAME} cuda_lib cuda_common)
target_compile_features(cuda_lib PUBLIC cxx_std_14)
set_target_properties(cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cuda_lib BEFORE
        PUBLIC
        /usr/local/cuda/include
)
target_include_directories(${PROJECT_NAME} BEFORE
        PUBLIC
        /usr/local/cuda/include
)
set(CMAKE_CXX_COMPILER /usr/bin/clang++-8)
set(CUDA_HOST_COMPILER /usr/bin/clang++-3.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")

