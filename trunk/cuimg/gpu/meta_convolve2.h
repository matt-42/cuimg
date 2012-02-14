#ifndef CUIMG_META_CONVOLVE2_STATIC_H_
# define CUIMG_META_CONVOLVE2_STATIC_H_

# include <cuimg/util.h>
# include <cuimg/error.h>
# include <cuimg/gpu/texture.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/gpu/meta_convolve.h>
# include <cuimg/gpu/image2d.h>
# include <cuimg/meta_gaussian/meta_gaussian.h>

namespace cuimg
{
  namespace meta_convolve_internal
  {

    template <typename I, int R, int E, typename G1, typename G2>
      struct meta_convolve2d_row_loop2
      {
        template <typename U>
        static __device__ inline void iter(U& r1, U& r2, const i_int2& p)
        {;
          U v = U(tex2D(UNIT_STATIC(g_input_tex)<typename U::cuda_bt>::tex(), p.y + R, p.x));
          r1 += v * meta_kernel<G1, R>::coef();
          r2 += v * meta_kernel<G2, R>::coef();

          meta_convolve2d_row_loop2<I, R + 1, E, G1, G2>::iter(r1, r2, p);
        }
      };

    template <typename I, int E, typename G1, typename G2>
    struct meta_convolve2d_row_loop2<I, E, E, G1, G2>
    {
      template <typename U>
        static __device__ inline void iter(U& r1, U& r2, const i_int2& p)
      {;
        U v = U(tex2D(UNIT_STATIC(g_input_tex)<typename U::cuda_bt>::tex(), p.y + E, p.x));
        r1 += v * meta_kernel<G1, E>::coef();
        r2 += v * meta_kernel<G2, E>::coef();
      }
    };

    template <typename I, typename O, typename G1, typename G2, int KERNEL_HALF_SIZE>
    static __global__ void meta_convolve_row_static_kernel2(kernel_image2d<O> out1,
                                                            kernel_image2d<O> out2)
    {
      int idr = blockIdx.y * blockDim.y + threadIdx.y;
      int idc = blockIdx.x * blockDim.x + threadIdx.x;
      i_int2 p(idr, idc);
      if (!out1.has(p))
        return;
      O r1 = zero();
      O r2 = zero();
      meta_convolve2d_row_loop2<I, -KERNEL_HALF_SIZE, KERNEL_HALF_SIZE, G1, G2>::iter(r1, r2, p);
      out1(p) = r1;
      out2(p) = r2;
    }

    template <typename I, int R, int E, typename G1, typename G2>
      struct meta_convolve2d_col_loop2
      {
        template <typename U>
        static __device__ inline void iter(U& r1, U& r2, const i_int2& p)
        {
          U v = U(tex2D(UNIT_STATIC(g_input_tex)<typename U::cuda_bt>::tex(), p.y, p.x + R));
          U v2 = U(tex2D(UNIT_STATIC(g_input_tex2)<typename U::cuda_bt>::tex(), p.y, p.x + R));
          r1 += v * meta_kernel<G1, R>::coef();
          r2 += v2 * meta_kernel<G2, R>::coef();
          meta_convolve2d_col_loop2<I, R + 1, E, G1, G2>::iter(r1, r2, p);
        }
      };

    template <typename I, int E, typename G1, typename G2>
    struct meta_convolve2d_col_loop2<I, E, E, G1, G2>
    {
      template <typename U>
      static __device__ inline void iter(U& r1, U& r2, const i_int2& p)
      {
        U v = U(tex2D(UNIT_STATIC(g_input_tex)<I>::tex(), p.y, p.x + E));
        U v2 = U(tex2D(UNIT_STATIC(g_input_tex2)<typename U::cuda_bt>::tex(), p.y, p.x + E));
        r1 += v * meta_kernel<G1, E>::coef();
        r2 += v2 * meta_kernel<G2, E>::coef();
      }
    };

    template <typename I, typename O, typename G1, typename G2, int KERNEL_HALF_SIZE>
    static __global__ void meta_convolve_col_static_kernel2(kernel_image2d<O> out1,
                                                           kernel_image2d<O> out2)
    {
      int idr = blockIdx.y * blockDim.y + threadIdx.y;
      int idc = blockIdx.x * blockDim.x + threadIdx.x;
      i_int2 p(idr, idc);
      if (!out1.has(p))
        return;
      O r1 = zero();
      O r2 = zero();
      meta_convolve2d_col_loop2<I, -KERNEL_HALF_SIZE, KERNEL_HALF_SIZE, G1, G2>::iter(r1, r2, p);
      out1(p) = r1;
      out2(p) = r2;
    }
  }

  template <typename I, typename O, typename G1, typename G2, int KERNEL_HALF_SIZE>
  void meta_convolve_row2d(const I& in, O& out1, O& out2,
                           cudaStream_t stream = 0, dim3 dimblock = dim3(128, 1))
  {
    assert(in.domain() == out.domain());
    bindTexture2d(in, meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);
    meta_convolve_internal::meta_convolve_row_static_kernel2
      <typename I::value_type::cuda_bt, typename O::value_type, G1, G2, KERNEL_HALF_SIZE>
      <<<dimgrid, dimblock, 0, stream>>>(mki(out1), mki(out2));
    cudaUnbindTexture(meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());
    check_cuda_error();
  }

  template <typename I, typename O, typename G1, typename G2, int KERNEL_HALF_SIZE>
  void meta_convolve_col2d(const I& in1, const I& in2, O& out1, O& out2,
                           cudaStream_t stream = 0, dim3 dimblock = dim3(128, 1))
  {
    assert(in.domain() == out.domain());
    bindTexture2d(in1, meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());
    bindTexture2d(in2, meta_convolve_internal::UNIT_STATIC(g_input_tex2)<typename I::value_type::cuda_bt>::tex());
    dim3 dimgrid = grid_dimension(in1.domain(), dimblock);
    meta_convolve_internal::meta_convolve_col_static_kernel2
      <typename I::value_type::cuda_bt, typename O::value_type, G1, G2, KERNEL_HALF_SIZE>
      <<<dimgrid, dimblock, 0, stream>>>(mki(out1), mki(out2));
    cudaUnbindTexture(meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());
    cudaUnbindTexture(meta_convolve_internal::UNIT_STATIC(g_input_tex2)<typename I::value_type::cuda_bt>::tex());
    check_cuda_error();
  }

}

#endif
