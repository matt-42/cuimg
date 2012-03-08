#ifndef CUIMG_IMAGE2D_MATH_H_
# define CUIMG_IMAGE2D_MATH_H_

# ifdef NVCC

# include <cuda.h>
# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/builtin_math.h>
# include <cuimg/util.h>

namespace cuimg
{

  namespace internal
  {
    template <typename I, typename O>
    __global__ void abs_kernel(kernel_image2d<I> in, kernel_image2d<O> out)
    {
      i_int2 p = thread_pos2d();

      if (out.has(p))
        out(p) += i_float4(0.5f,0.5f,0.5f,0);
    }

    template <typename I> __device__ I alpha_max();

    template <> __device__ inline float alpha_max<float>() { return 1.; }
    template <> __device__ inline char alpha_max<char>() { return 127; }
    template <> __device__ inline unsigned char alpha_max<unsigned char>() { return 255; }

    template <typename I>
    __global__ void set_alpha_kernel(kernel_image2d<I> img)
    {
      i_int2 p = thread_pos2d();

      if (img.has(p))
        img(p).w = alpha_max<bt_vtype(I)>();
    }

    template <typename I, typename S>
    __global__ void mult_kernel(kernel_image2d<I> img, const S s)
    {
      i_int2 p = thread_pos2d();

      if (img.has(p))
//        img(p) *= s;
        img(p) = img(p) * s;
    }

    template <typename I>
    __global__ void add_kernel(kernel_image2d<I> img, const I a)
    {
      i_int2 p = thread_pos2d();

      if (img.has(p))
        img(p) += a;
    }

    template <typename I>
    __global__ void minus_kernel(kernel_image2d<I> a, kernel_image2d<I> b, kernel_image2d<I> out)
    {
      i_int2 p = thread_pos2d();

      if (out.has(p))
        out(p) = a(p) - b(p);
    }

    template <typename I, unsigned N>
    __global__ void unsaturate_kernel(kernel_image2d<improved_builtin<I, N> > img, I v)
    {
      i_int2 p = thread_pos2d();

      if (img.has(p))
      {
        improved_builtin<I, N> res = img(p);
        for (unsigned i = 0; i < N; i++)
          res[i] = res[i] > v ? v : res[i];
        img(p) = res;
      }
    }

  }

  template <typename I, typename O>
  inline
  void abs(const image2d<I>& in, const image2d<O>& out, dim3 dimblock = dim3(16, 16))
  {
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);
    internal::abs_kernel<<<dimgrid, dimblock>>>(mki(in), mki(out));
  }

  template <typename I>
  inline
  void set_alpha_channel(image2d<I>& in, dim3 dimblock = dim3(16, 16))
  {
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);
    internal::set_alpha_kernel<<<dimgrid, dimblock>>>(mki(in));
  }


  template <typename I, typename S>
  inline
  void mult(image2d<I>& in, const S& s, dim3 dimblock = dim3(16, 16))
  {
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);
    internal::mult_kernel<<<dimgrid, dimblock>>>(mki(in), s);
  }

  template <typename I>
  inline
  void add(image2d<I>& in, const I& s, dim3 dimblock = dim3(16, 16))
  {
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);
    internal::add_kernel<<<dimgrid, dimblock>>>(mki(in), s);
  }

  template <typename A,
            typename B,
            typename O>
  inline
  void minus(image2d<A>& a, image2d<B>& b, image2d<O>& out, dim3 dimblock = dim3(16, 16))
  {
    dim3 dimgrid = grid_dimension(a.domain(), dimblock);
    internal::minus_kernel<<<dimgrid, dimblock>>>(mki(a), mki(b), mki(out));
  }

  template <typename A, typename V>
  inline
  void unsaturate(image2d<A>& a, V v, dim3 dimblock = dim3(16, 16))
  {
    dim3 dimgrid = grid_dimension(a.domain(), dimblock);
    internal::unsaturate_kernel<<<dimgrid, dimblock>>>(mki(a), v);
  }


}

#endif

#endif
