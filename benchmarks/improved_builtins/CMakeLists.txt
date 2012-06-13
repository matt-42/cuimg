cmake_minimum_required (VERSION 2.6)
project (bench)

find_package(Cuda)

include_directories(${CMAKE_SOURCE_DIR}/../.. ${CMAKE_SOURCE_DIR}/../../../../programs/boost_1_42_0)

SET(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/cuda_tmp)
SET(CUDA_VERBOSE_BUILD ON)
SET(CUDA_BUILD_CUBIN ON)

CUDA_ADD_EXECUTABLE(bench main.cu OPTIONS -keep)

target_link_libraries(bench ${CUDA_LIBRARIES})
