cmake_minimum_required (VERSION 2.6)

project (tracking_sample)

find_package(OpenCV REQUIRED)

ADD_DEFINITIONS(-std=c++0x)
#ADD_DEFINITIONS(-g)
ADD_DEFINITIONS(-std=c++0x -fopenmp -DNDEBUG -DNO_CUDA -DWITH_OPENCV -Ofast -O3 -funroll-loops -march=native)

include_directories(${CMAKE_SOURCE_DIR}/../..)
SET(sources main.cc)
SET(CMAKE_VERBOSE_MAKEFILE ON)

ADD_EXECUTABLE(tracking_sample ${sources})
target_link_libraries(tracking_sample ${OpenCV_LIBS} gomp)
