#ifndef CUIMG_IMAGE2D_MATH_H_
# define CUIMG_IMAGE2D_MATH_H_

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

    template <> __device__ float alpha_max<float>() { return 1.; }
    template <> __device__ char alpha_max<char>() { return 127; }
    template <> __device__ unsigned char alpha_max<unsigned char>() { return 255; }

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


}

#endif
