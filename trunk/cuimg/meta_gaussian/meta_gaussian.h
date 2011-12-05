#ifndef CUIMG_META_GAUSSIAN_COMMON_H_
# define CUIMG_META_GAUSSIAN_COMMON_H_

namespace cuimg
{
  template <int N, int S>
  struct meta_gaussian
  {
  };

  template <typename T, int X>
  struct meta_kernel
  {
    static __host__ __device__ float coef() {return 0.f;}
  };

}

#endif
