#ifndef CUIMG_ERROR_H_
# define CUIMG_ERROR_H_


#ifndef NO_CUDA

# include <cuimg/gpu/cuda.h>
# include <iostream>
# include <cassert>
# include <stdexcept>
#endif

namespace cuimg
{


  inline void check_cuda_error()
  {
# ifndef NO_CUDA
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      std::cerr << "Cuda error: " << cudaGetErrorString(error) << std::endl;
      throw std::runtime_error(std::string(cudaGetErrorString(error)));
      exit(1);
      assert(error == cudaSuccess);
    }
# endif
  }

}


#endif
