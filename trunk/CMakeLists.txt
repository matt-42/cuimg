cmake_minimum_required (VERSION 2.8)
project (cuimg)

find_package(Cuda REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})

SET(SOURCES cuimg/video_capture.cpp)

ADD_LIBRARY(cuimg ${SOURCES})

ENABLE_TESTING()


CUDA_ADD_EXECUTABLE(improved_builtin tests/improved_builtin.cu)
TARGET_LINK_LIBRARIES(improved_builtin ${CUDA_LIBRARIES})
ADD_TEST(improved_builtin ${EXECUTABLE_OUTPUT_PATH}/improved_builtin)

CUDA_ADD_EXECUTABLE(image2d tests/image2d.cu)
TARGET_LINK_LIBRARIES(image2d ${CUDA_LIBRARIES})
ADD_TEST(image2d ${EXECUTABLE_OUTPUT_PATH}/image2d)

CUDA_ADD_EXECUTABLE(host_image3d tests/host_image3d.cpp)
TARGET_LINK_LIBRARIES(host_image3d ${CUDA_LIBRARIES})
ADD_TEST(host_image3d ${EXECUTABLE_OUTPUT_PATH}/host_image3d)

CUDA_ADD_EXECUTABLE(shared_ptr tests/shared_ptr.cu)
TARGET_LINK_LIBRARIES(shared_ptr)
ADD_TEST(shared_ptr ${EXECUTABLE_OUTPUT_PATH}/shared_ptr)

CUDA_ADD_EXECUTABLE(point2d tests/point2d.cu)
TARGET_LINK_LIBRARIES(point2d)
ADD_TEST(point2d ${EXECUTABLE_OUTPUT_PATH}/point2d)

CUDA_ADD_EXECUTABLE(obox2d tests/obox2d.cu)
TARGET_LINK_LIBRARIES(obox2d)
ADD_TEST(obox2d ${EXECUTABLE_OUTPUT_PATH}/obox2d)
