#ifndef CUIMG_META_GAUSSIAN_COMMON_H_
# define CUIMG_META_GAUSSIAN_COMMON_H_

namespace cuimg
{
  template <int N, int S>
  struct meta_gaussian
  {
    enum { n = N };
    enum { s = S };
  };

  template <typename T, int X>
  struct meta_kernel
  {
    enum { s, n, x};
    static __host__ __device__ float coef() {return 0.f;}
  };

}

#endif
