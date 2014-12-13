
CUIMG An Image and Video Processing Library
================================

Note: This project is not active anymore. It has been rewritten with c++14 (and without
CUDA since it does not support C++14). Check the the Video++ library: http://github.com/matt-42/vpp

CUIMG is a library for image and video processing. It is released under the GPLv3 licence.
It targets GPUs with CUDA, and CPUs with the C++0x language.

Its main features are:
    - Basic 2D and 3D image types
    - A fast semi dense points tracker
    - Interoperability with OpenCV image types.

An example to use the semi dense tracker is available here:
https://github.com/matt-42/cuimg/tree/master/samples/tracking


Dependencies:
    - Boost
    - OpenCV
    - Cuda (optional, disable it with -DNO_CUDA )
