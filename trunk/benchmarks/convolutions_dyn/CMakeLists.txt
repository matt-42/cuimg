cmake_minimum_required (VERSION 2.6)
project (convolutions_dyn)

find_package(Cuda)

include_directories(${CMAKE_SOURCE_DIR}/../.. ${CMAKE_SOURCE_DIR}/../../../../programs/boost_1_42_0)

SET(CUDA_VERBOSE_BUILD ON)
SET(CUDA_BUILD_CUBIN ON)

CUDA_ADD_EXECUTABLE(convolutions_dyn main.cu OPTIONS -keep --gpu-architecture sm_13)

target_link_libraries(convolutions_dyn ${CUDA_LIBRARIES})

