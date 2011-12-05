#ifndef CUIMG_BUILTIN_MATH_H_
# define CUIMG_BUILTIN_MATH_H_

# include <cuda.h>
# include <cuimg/improved_builtin.h>
# include <cuimg/meta.h>

namespace cuimg
{

  namespace internal
  {
    // --------------------------- Unrolling operators ---------------------------------
    template <int I>
    class abs
    {
    public:
      template <typename U, typename V, unsigned US, unsigned VS>
        static __device__ __host__ inline void run(improved_builtin<U, US>& a, const improved_builtin<V, VS>& b)
      {
        bt_getter<I>::get(a) =  ::abs(bt_getter<I>::get(b));
      }
    };

    template <int I>
    class sqrt
    {
    public:
      template <typename U, typename V, unsigned US, unsigned VS>
        static __device__ __host__ inline void run(improved_builtin<U, US>& a, const improved_builtin<V, VS>& b)
      {
        bt_getter<I>::get(a) =  ::sqrt(bt_getter<I>::get(b));
      }
    };

    template <int I>
    class norml2
    {
    public:
      template <typename U, typename V, unsigned VS>
        static __device__ __host__ inline void run(U& out, const improved_builtin<V, VS>& in)
      {
        out += bt_getter<I>::get(in) * bt_getter<I>::get(in);
      }
    };

    template <int I>
    class norml1
    {
    public:
      template <typename U, typename V, unsigned VS>
        static __device__ __host__ inline void run(U& out, const improved_builtin<V, VS>& in)
      {
        out += ::abs(bt_getter<I>::get(in));
      }
    };

    template <int I>
    class norminf
    {
    public:
      template <typename U, typename V, unsigned VS>
        static __device__ __host__ inline void run(U& out, const improved_builtin<V, VS>& in)
      {
        out = out >= bt_getter<I>::get(in) ? out : bt_getter<I>::get(in);
      }
    };

  }

  template <typename T, unsigned N>
  __host__ __device__ inline
  improved_builtin<T, N> abs(const improved_builtin<T, N>& x)
  {
    improved_builtin<T, N> r;
    meta::loop<internal::abs, 0, N - 1>::iter(r, x);
    return r;
  }

  template <typename T, unsigned N>
  __host__ __device__ inline
  improved_builtin<T, N> sqrt(const improved_builtin<T, N>& x)
  {
    improved_builtin<T, N> r;
    meta::loop<internal::sqrt, 0, N - 1>::iter(r, x);
    return r;
  }

    template <typename T, unsigned N>
  __host__ __device__ inline
  float norml1(const improved_builtin<T, N>& x)
  {
    float r = 0;
    meta::loop<internal::norml1, 0, N - 1>::iter(r, x);
    return r;
  }

  template <typename T, unsigned N>
  __host__ __device__ inline
  float norml2(const improved_builtin<T, N>& x)
  {
    float r = 0;
    meta::loop<internal::norml2, 0, N - 1>::iter(r, x);
    return ::sqrt(r);
  }

  template <typename T, unsigned N>
  __host__ __device__ inline
  float norminf(const improved_builtin<T, N>& x)
  {
    float r = 0;
    meta::loop<internal::norminf, 0, N - 1>::iter(r, x);
    return r;
  }


}

#endif
