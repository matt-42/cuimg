#ifndef CUIMG_META_GAUSSIAN_SIGMA_0_H_
# define CUIMG_META_GAUSSIAN_SIGMA_0_H_


namespace cuimg
{

  template <int N, int S, int X> struct meta_gaussian_coef;
  template <int N, int X> struct meta_gaussian_coef<N, 0, X> { static __device__ __host__ inline float coef() { return float(0); } };

  // 0 th derivative.
  template <> struct meta_gaussian_coef<0, 0, 0> { static __device__ __host__ inline float coef() { return float(1.f); } };
 
  // 1 th derivative.
  template <> struct meta_gaussian_coef<1, 0, -1> { static __device__ __host__ inline float coef() { return float(1.f); } };
  template <> struct meta_gaussian_coef<1, 0, 0> { static __device__ __host__ inline float coef() { return float(0.f); } };
  template <> struct meta_gaussian_coef<1, 0, 1> { static __device__ __host__ inline float coef() { return float(-1.f); } };

   // 2 th derivative.
  template <> struct meta_gaussian_coef<2, 0, -1> { static __device__ __host__ inline float coef() { return float(1.f); } };
  template <> struct meta_gaussian_coef<2, 0, 0> { static __device__ __host__ inline float coef() { return float(-2.f); } };
  template <> struct meta_gaussian_coef<2, 0, 1> { static __device__ __host__ inline float coef() { return float(1.f); } };

} // end of namespace cuimg.
#endif
