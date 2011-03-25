#ifndef CUIMG_ERROR_H_
# define CUIMG_ERROR_H_

# include <cuda_runtime.h>
# include <iostream>
# include <boost/shared_ptr.hpp>
# include <cuimg/point2d.h>

namespace cuimg
{


  inline void check_cuda_error()
  {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      std::cerr << "Cuda error: " << cudaGetErrorString(error) << std::endl;
      assert(error == cudaSuccess);
    }
  }

}

#endif
