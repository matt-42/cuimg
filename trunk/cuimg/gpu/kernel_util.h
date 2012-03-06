#ifndef CUIMG_GPU_KERNEL_UTIL_H_
# define CUIMG_GPU_KERNEL_UTIL_H_

# include <cuda.h>
# include <cuimg/obox2d.h>
# include <cuimg/obox3d.h>
# include <cuimg/util.h>
# include <cuimg/pw_call.h>

namespace cuimg
{
  __host__ __device__ inline i_int2 thread_pos2d();
  __host__ __device__ inline i_int3 thread_pos3d();

  template <unsigned T>
  __host__ __device__ inline i_int2 thread_pos2d(const thread_info<T>& ti)
  {
    return i_int2(ti.blockIdx.y * ti.blockDim.y + ti.threadIdx.y,
                  ti.blockIdx.x * ti.blockDim.x + ti.threadIdx.x);
  }

#ifdef NVCC
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
#endif

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

  template <typename T>
  __host__ __device__ inline void cuswap(T& a, T& b)
  {
    T tmp = a;
    a = b;
    b = tmp;
  }

}

#endif
