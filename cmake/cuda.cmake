
set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}")
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)

find_library(CUDART_LIB cudart
        HINTS
        "${CUDA_TOOLKIT_ROOT_DIR}/../lib64"
        "${CUDA_TOOLKIT_ROOT_DIR}/../lib"
        "${CUDA_TOOLKIT_ROOT_DIR}"
        )

add_library(CUDA::cudart IMPORTED INTERFACE)
set_target_properties(CUDA::cudart PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        INTERFACE_LINK_LIBRARIES
        "${CUDART_LIB}")

find_library(CUDA_LIB cuda
        HINTS
        "${CUDA_TOOLKIT_ROOT_DIR}/../lib64"
        "${CUDA_TOOLKIT_ROOT_DIR}/../lib"
        "${CUDA_TOOLKIT_ROOT_DIR}"
        )

add_library(CUDA::cuda IMPORTED INTERFACE)
set_target_properties(CUDA::cuda PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        INTERFACE_LINK_LIBRARIES
        "${CUDA_LIB}")


find_library(CUDNN_LIB cudnn
        HINTS
        "${CUDA_TOOLKIT_ROOT_DIR}/../lib64"
        "${CUDA_TOOLKIT_ROOT_DIR}/../lib"
        "${CUDA_TOOLKIT_ROOT_DIR}"
        )
add_library(CUDA::cudnn IMPORTED INTERFACE)
set_target_properties(CUDA::cudnn PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        INTERFACE_LINK_LIBRARIES
        "${CUDNN_LIB}")

find_library(CUBLAS_LIB cublas
        HINTS
        "${CUDA_TOOLKIT_ROOT_DIR}/../lib64"
        "${CUDA_TOOLKIT_ROOT_DIR}/../lib"
        "${CUDA_TOOLKIT_ROOT_DIR}"
        )
add_library(CUDA::cublas IMPORTED INTERFACE)
set_target_properties(CUDA::cublas PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        INTERFACE_LINK_LIBRARIES
        "${CUBLAS_LIB}")
