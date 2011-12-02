#ifndef CUIMG_META_CONVOLVE_STATIC_H_
# define CUIMG_META_CONVOLVE_STATIC_H_

# include <cuimg/util.h>
# include <cuimg/error.h>
# include <cuimg/gpu/texture.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/gpu/image2d.h>
# include <cuimg/meta_gaussian/meta_gaussian.h>

namespace cuimg
{
  namespace meta_convolve_internal
  {

    template <typename T>
    struct UNIT_STATIC(g_input_tex);
    REGISTER_TEXTURE2D_PROXY(g_input_tex);

    template <typename I, int R, int E, typename G>
      struct meta_convolve2d_row_loop
      {
        template <typename U>
        static __device__ inline U iter(const kernel_image2d<U>& out, const i_int2& p)
        {;
          return U(tex2D(UNIT_STATIC(g_input_tex)<typename U::cuda_bt>::tex(), p.y + R, p.x)) * meta_kernel<G, R>::coef() +
            meta_convolve2d_row_loop<I, R + 1, E, G>::iter(out, p);
        }
      };

    template <typename I, int E, typename G>
      struct meta_convolve2d_row_loop<I, E, E, G>
    {
      template <typename U>
        static __device__ inline U iter(const kernel_image2d<U>& out, const i_int2& p)
      {;
        return U(tex2D(UNIT_STATIC(g_input_tex)<I>::tex(), p.y + E, p.x)) * meta_kernel<G, E>::coef();
      }
    };

    template <typename I, typename O, typename G, int KERNEL_HALF_SIZE>
    static __global__ void meta_convolve_row_static_kernel(kernel_image2d<O> out)
    {
      int idr = blockIdx.y * blockDim.y + threadIdx.y;
      int idc = blockIdx.x * blockDim.x + threadIdx.x;
      i_int2 p(idr, idc);
      if (!out.has(p))
        return;
      out(p) = meta_convolve2d_row_loop<I, -KERNEL_HALF_SIZE, KERNEL_HALF_SIZE, G>::iter(out, p);
    }

    template <typename I, int R, int E, typename G>
      struct meta_convolve2d_col_loop
      {
        template <typename U>
        static __device__ inline U iter(const kernel_image2d<U>& out, const i_int2& p)
        {;
          return U(tex2D(UNIT_STATIC(g_input_tex)<typename U::cuda_bt>::tex(), p.y, p.x + R)) * meta_kernel<G, R>::coef() +
            meta_convolve2d_col_loop<I, R + 1, E, G>::iter(out, p);
        }
      };

    template <typename I, int E, typename G>
      struct meta_convolve2d_col_loop<I, E, E, G>
    {
      template <typename U>
        static __device__ inline U iter(const kernel_image2d<U>& out, const i_int2& p)
      {;
        return U(tex2D(UNIT_STATIC(g_input_tex)<I>::tex(), p.y, p.x + E)) * meta_kernel<G, E>::coef();
      }
    };

    template <typename I, typename O, typename G, int KERNEL_HALF_SIZE>
    static __global__ void meta_convolve_col_static_kernel(kernel_image2d<O> out)
    {
      int idr = blockIdx.y * blockDim.y + threadIdx.y;
      int idc = blockIdx.x * blockDim.x + threadIdx.x;
      i_int2 p(idr, idc);
      if (!out.has(p))
        return;
      out(p) = meta_convolve2d_col_loop<I, -KERNEL_HALF_SIZE, KERNEL_HALF_SIZE, G>::iter(out, p);
    }
  }

  template <typename I, typename O, typename G, int KERNEL_HALF_SIZE>
  void meta_convolve_row2d(const I& in, O& out,
                           cudaStream_t stream = 0, dim3 dimblock = dim3(16, 16))
  {
    assert(in.domain() == out.domain());
    bindTexture2d(in, meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);
    meta_convolve_internal::meta_convolve_row_static_kernel
    <typename I::value_type::cuda_bt, typename O::value_type, G, KERNEL_HALF_SIZE>
      <<<dimgrid, dimblock, 0, stream>>>(mki(out));
    cudaUnbindTexture(meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());
    check_cuda_error();
  }

  template <typename I, typename O, typename G, int KERNEL_HALF_SIZE>
  void meta_convolve_col2d(const I& in, O& out,
                           cudaStream_t stream = 0, dim3 dimblock = dim3(16, 16))
  {
    assert(in.domain() == out.domain());
    bindTexture2d(in, meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);
    meta_convolve_internal::meta_convolve_col_static_kernel
    <typename I::value_type::cuda_bt, typename O::value_type, G, KERNEL_HALF_SIZE>
      <<<dimgrid, dimblock, 0, stream>>>(mki(out));
    cudaUnbindTexture(meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());
    check_cuda_error();
  }

}

#endif
