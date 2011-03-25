#ifndef CUIMG_UTIL_H_
# define CUIMG_UTIL_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/obox2d.h>
# include <cuimg/obox3d.h>
# include <cuimg/error.h>

namespace cuimg
{
  __host__ __device__ inline i_int2 thread_pos2d()
  {
    return i_int2(blockIdx.y * blockDim.y + threadIdx.y,
                  blockIdx.x * blockDim.x + threadIdx.x);
  }

  __host__ __device__ inline i_int3 thread_pos3d()
  {
    return i_int3(blockIdx.z * blockDim.z + threadIdx.z,
                  blockIdx.y * blockDim.y + threadIdx.y,
                  blockIdx.x * blockDim.x + threadIdx.x);
  }

  template <typename T>
  inline dim3 grid_dimension(const obox2d<T>& domain, const dim3& dimblock)
  {
    return dim3(idivup(domain.ncols(), dimblock.x), idivup(domain.nrows(), dimblock.y));
  }

  template <typename T>
  inline dim3 grid_dimension(const obox3d<T>& domain, const dim3& dimblock)
  {
    return dim3(idivup(domain.ncols(),   dimblock.x),
                idivup(domain.nrows(),   dimblock.y),
                idivup(domain.nslices(), dimblock.z));
  }

  #define CUIMG_PI 3.14159265
}

#endif
