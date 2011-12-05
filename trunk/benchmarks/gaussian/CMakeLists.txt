cmake_minimum_required (VERSION 2.6)
project (gaussian)

find_package(Cuda)

include_directories(${CMAKE_SOURCE_DIR}/../.. ${CMAKE_SOURCE_DIR}/../../../../programs/boost_1_42_0)

SET(CUDA_VERBOSE_BUILD ON)

SET(CUDA_BUILD_CUBIN OFF)

CUDA_ADD_EXECUTABLE(gaussian main.cu OPTIONS -keep --gpu-architecture sm_10)

target_link_libraries(gaussian ${CUDA_LIBRARIES})

