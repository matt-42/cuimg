
cmake_minimum_required (VERSION 2.8)
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake
		      ${CMAKE_MODULE_PATH}
                      )

project (cuimg)


find_package(CUDA)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})

#ADD_DEFINITIONS(-std=c++0x -DNDEBUG -O3 -fopenmp -march=native -funroll-loops -DWITH_OPENCV)
add_definitions(-std=c++0x -g  -DWITH_OPENCV -DNO_CUDA)

ADD_DEFINITIONS(-DNO_CUDA)

IF (NOT CUDA_FOUND)
  add_definitions(-DNO_CUDA)
  set(CUDA_LIBRARIES "")
#  ADD_EXECUTABLE(image2d tests/image2d.cu)
#  ADD_EXECUTABLE(host_image3d tests/host_image3d.cpp)
#  ADD_EXECUTABLE(shared_ptr tests/shared_ptr.cu)
#  ADD_EXECUTABLE(point2d tests/point2d.cu)
#  ADD_EXECUTABLE(obox2d tests/obox2d.cu)
#  ADD_EXECUTABLE(make_meta_gaussian make_meta_gaussian.cpp)
#  ADD_EXECUTABLE(improved_builtin tests/improved_builtin.cu)

#  SET_TARGET_PROPERTIES(image2d PROPERTIES LINKER_LANGUAGE C)
#  SET_TARGET_PROPERTIES(host_image3d PROPERTIES LINKER_LANGUAGE C)
#  SET_TARGET_PROPERTIES(shared_ptr PROPERTIES LINKER_LANGUAGE C)
#  SET_TARGET_PROPERTIES(point2d PROPERTIES LINKER_LANGUAGE C)
#  SET_TARGET_PROPERTIES(obox2d PROPERTIES LINKER_LANGUAGE C)
#  SET_TARGET_PROPERTIES(make_meta_gaussian PROPERTIES LINKER_LANGUAGE C)
#  SET_TARGET_PROPERTIES(improved_builtin PROPERTIES LINKER_LANGUAGE C)

ELSE (NOT CUDA_FOUND)
  Message("CUDA_FOUND")
  SET(CUDA_VERBOSE_BUILD ON)
  SET(BUILD_SHARED_LIBS OFF)
  CUDA_ADD_EXECUTABLE(image2d tests/image2d.cu)
  CUDA_ADD_EXECUTABLE(host_image3d tests/host_image3d.cpp)
  CUDA_ADD_EXECUTABLE(shared_ptr tests/shared_ptr.cu)
  CUDA_ADD_EXECUTABLE(point2d tests/point2d.cu)
  CUDA_ADD_EXECUTABLE(obox2d tests/obox2d.cu)
  CUDA_ADD_EXECUTABLE(make_meta_gaussian make_meta_gaussian.cpp)
  CUDA_ADD_EXECUTABLE(improved_builtin tests/improved_builtin.cu)

  set(CUDA_SEPARABLE_COMPILATION ON)
  SET(CUDA_NVCC_FLAGS "--pre-include ${CMAKE_SOURCE_DIR}/cuimg/gpu/undef_atomics_int128.h;-Xcompiler -DNO_CPP0X")

  CUDA_ADD_LIBRARY(cuimg_cuda cuimg/tracking2/bc2s_feature.cu STATIC OPTIONS -arch sm_20)

ENDIF (NOT CUDA_FOUND)

SET(SOURCES cuimg/video_capture.cpp
            cuimg/background_video_capture.cpp
            cuimg/load_image.cpp
            cuimg/save_image.cpp
#	    cuimg/tracking2/tracker.cpp
#	    cuimg/tracking2/opencv_tracker.cpp
	    cuimg/tracking2/opencv_klttracker.cpp
            )

ADD_LIBRARY(cuimg ${SOURCES})

install(TARGETS cuimg DESTINATION lib)
install(TARGETS cuimg DESTINATION lib/Debug CONFIGURATIONS Debug)
install(TARGETS cuimg DESTINATION lib/Release CONFIGURATIONS Release)

install(DIRECTORY cuimg DESTINATION include
  FILES_MATCHING PATTERN "*.h")
install(DIRECTORY cuimg DESTINATION include
  FILES_MATCHING PATTERN "*.cu")
install(DIRECTORY cuimg DESTINATION include
  FILES_MATCHING PATTERN "*.hpp")


ENABLE_TESTING()


TARGET_LINK_LIBRARIES(improved_builtin ${CUDA_LIBRARIES})
ADD_TEST(improved_builtin ${EXECUTABLE_OUTPUT_PATH}/improved_builtin)

TARGET_LINK_LIBRARIES(image2d ${CUDA_LIBRARIES})
ADD_TEST(image2d ${EXECUTABLE_OUTPUT_PATH}/image2d)

TARGET_LINK_LIBRARIES(host_image3d ${CUDA_LIBRARIES})
ADD_TEST(host_image3d ${EXECUTABLE_OUTPUT_PATH}/host_image3d)

TARGET_LINK_LIBRARIES(shared_ptr)
ADD_TEST(shared_ptr ${EXECUTABLE_OUTPUT_PATH}/shared_ptr)

TARGET_LINK_LIBRARIES(point2d)
ADD_TEST(point2d ${EXECUTABLE_OUTPUT_PATH}/point2d)

TARGET_LINK_LIBRARIES(obox2d)
ADD_TEST(obox2d ${EXECUTABLE_OUTPUT_PATH}/obox2d)










