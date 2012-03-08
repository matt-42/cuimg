#ifndef CUIMG_META_CONVOLVE2_STATIC_H_
# define CUIMG_META_CONVOLVE2_STATIC_H_

# include <cuimg/util.h>
# include <cuimg/error.h>
# include <cuimg/gpu/texture.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/gpu/meta_convolve.h>
# include <cuimg/gpu/device_image2d.h>
# include <cuimg/meta_gaussian/meta_gaussian.h>

namespace cuimg
{
  namespace meta_convolve_internal
  {

    template <typename I, int R, int E, typename G1, typename G2>
    struct meta_convolve2d_row_loop2
    {
      template <unsigned T, typename U>
      static __host__ __device__ inline void iter(const kernel_image2d<U>& in, U& r1, U& r2, const i_int2& p)
      {;
        U v = U(tex2D(flag<T>(), UNIT_STATIC(g_input_tex)<typename U::cuda_bt>::tex(), in, p.y + R, p.x));
        r1 += v * meta_kernel<G1, R>::coef();
        r2 += v * meta_kernel<G2, R>::coef();

        meta_convolve2d_row_loop2<I, R + 1, E, G1, G2>::template iter<T, U>(in, r1, r2, p);
      }
    };

    template <typename I, int E, typename G1, typename G2>
    struct meta_convolve2d_row_loop2<I, E, E, G1, G2>
    {
      template <unsigned T, typename U>
      static __host__ __device__ inline void iter(const kernel_image2d<U>& in, U& r1, U& r2, const i_int2& p)
      {;
        U v = U(tex2D(flag<T>(), UNIT_STATIC(g_input_tex)<typename U::cuda_bt>::tex(), in, p.y + E, p.x));
        r1 += v * meta_kernel<G1, E>::coef();
        r2 += v * meta_kernel<G2, E>::coef();
      }
    };

#define meta_convolve_row_static_kernel2_sig(T, I, O, G1, G2, KERNEL_HALF_SIZE) \
      kernel_image2d<O>,                                           \
    kernel_image2d<O> ,                                             \
      kernel_image2d<O>,                                           \
      &meta_convolve_internal::meta_convolve_row_static_kernel2<T, I, O, G1, G2, KERNEL_HALF_SIZE>

    template <unsigned T, typename I, typename O, typename G1, typename G2, int KERNEL_HALF_SIZE>
    __host__ __device__ void meta_convolve_row_static_kernel2(thread_info<T> ti,
                                                                     kernel_image2d<O> in1,
                                                                     kernel_image2d<O> out1,
                                                                     kernel_image2d<O> out2)
    {
      int idr = ti.blockIdx.y * ti.blockDim.y + ti.threadIdx.y;
      int idc = ti.blockIdx.x * ti.blockDim.x + ti.threadIdx.x;
      i_int2 p(idr, idc);
      if (!out1.has(p))
        return;
      if (T == CPU && (p.x < KERNEL_HALF_SIZE || p.y < KERNEL_HALF_SIZE ||
                       p.x >= in1.nrows() - KERNEL_HALF_SIZE || p.y >= in1.ncols() - KERNEL_HALF_SIZE))
        return;

      O r1 = zero();
      O r2 = zero();
      meta_convolve2d_row_loop2<I, -KERNEL_HALF_SIZE, KERNEL_HALF_SIZE, G1, G2>::template iter<T, O>(in1, r1, r2, p);
      out1(p) = r1;
      out2(p) = r2;
    }

    template <typename I, int R, int E, typename G1, typename G2>
    struct meta_convolve2d_col_loop2
    {
      template <unsigned T, typename U>
      static __host__ __device__ inline void iter(const kernel_image2d<U>& in1,
                                                  const kernel_image2d<U>& in2,
                                                  U& r1, U& r2, const i_int2& p)
      {
        U v = U(tex2D(flag<T>(), UNIT_STATIC(g_input_tex)<typename U::cuda_bt>::tex(), in1, p.y, p.x + R));
        U v2 = U(tex2D(flag<T>(), UNIT_STATIC(g_input_tex2)<typename U::cuda_bt>::tex(), in2, p.y, p.x + R));
        r1 += v * meta_kernel<G1, R>::coef();
        r2 += v2 * meta_kernel<G2, R>::coef();
        meta_convolve2d_col_loop2<I, R + 1, E, G1, G2>::template iter<T, U>(in1, in2, r1, r2, p);
      }
    };

    template <typename I, int E, typename G1, typename G2>
    struct meta_convolve2d_col_loop2<I, E, E, G1, G2>
    {
      template <unsigned T, typename U>
      static __host__ __device__ inline void iter(const kernel_image2d<U>& in1,
                                                  const kernel_image2d<U>& in2,
                                                  U& r1, U& r2, const i_int2& p)
      {
        U v = U(tex2D(flag<T>(), UNIT_STATIC(g_input_tex)<I>::tex(), in1, p.y, p.x + E));
        U v2 = U(tex2D(flag<T>(), UNIT_STATIC(g_input_tex2)<typename U::cuda_bt>::tex(), in2, p.y, p.x + E));
        r1 += v * meta_kernel<G1, E>::coef();
        r2 += v2 * meta_kernel<G2, E>::coef();
      }
    };

    template <unsigned T, typename I, typename O, typename G1, typename G2, int KERNEL_HALF_SIZE>
    __host__ __device__ void meta_convolve_col_static_kernel2(thread_info<T> ti,
                                                                     kernel_image2d<O> in1,
                                                                     kernel_image2d<O> in2,
                                                                     kernel_image2d<O> out1,
                                                                     kernel_image2d<O> out2)
    {
      int idr = ti.blockIdx.y * ti.blockDim.y + ti.threadIdx.y;
      int idc = ti.blockIdx.x * ti.blockDim.x + ti.threadIdx.x;
      i_int2 p(idr, idc);
      if (!out1.has(p))
        return;
      if (T == CPU && (p.x < KERNEL_HALF_SIZE || p.y < KERNEL_HALF_SIZE ||
                       p.x >= in1.nrows() - KERNEL_HALF_SIZE || p.y >= in1.ncols() - KERNEL_HALF_SIZE))
        return;

      O r1 = zero();
      O r2 = zero();
      meta_convolve2d_col_loop2<I, -KERNEL_HALF_SIZE, KERNEL_HALF_SIZE, G1, G2>::template iter<T, O>(in1, in2, r1, r2, p);
      out1(p) = r1;
      out2(p) = r2;
    }
  }

#define meta_convolve_col_static_kernel2_sig(T, I, O, G1, G2, KERNEL_HALF_SIZE) \
    kernel_image2d<O> ,                                             \
      kernel_image2d<O> ,                                           \
    kernel_image2d<O> ,                                             \
      kernel_image2d<O> ,                                           \
      &meta_convolve_internal::meta_convolve_col_static_kernel2<T, I, O, G1, G2, KERNEL_HALF_SIZE>

  template <typename I, typename O, typename G1, typename G2, int KERNEL_HALF_SIZE>
  void meta_convolve_row2d(const I& in, O& out1, O& out2,
                           cudaStream_t stream = 0, dim3 dimblock = dim3(128, 1))
  {
    assert(in.domain() == out1.domain());

    if (I::target == GPU)
      bindTexture2d(in, meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());

    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    pw_call<meta_convolve_row_static_kernel2_sig(I::target, typename I::value_type::cuda_bt,
                                                 typename O::value_type, G1, G2, KERNEL_HALF_SIZE)>
      (flag<I::target>(), dimgrid, dimblock, in, out1, out2);

    // meta_convolve_internal::meta_convolve_row_static_kernel2
    //   <typename I::value_type::cuda_bt, typename O::value_type, G1, G2, KERNEL_HALF_SIZE>
    //   <<<dimgrid, dimblock, 0, stream>>>(mki(out1), mki(out2));

    if (I::target == GPU)
    {
      cudaUnbindTexture(meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());
      check_cuda_error();
    }
  }

  template <typename I, typename O, typename G1, typename G2, int KERNEL_HALF_SIZE>
  void meta_convolve_col2d(const I& in1, const I& in2, O& out1, O& out2,
                           cudaStream_t stream = 0, dim3 dimblock = dim3(128, 1))
  {
    assert(in1.domain() == out1.domain());
    assert(in1.domain() == out2.domain());
    assert(in1.domain() == in2.domain());
    if (I::target == GPU)
    {
      bindTexture2d(in1, meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());
      bindTexture2d(in2, meta_convolve_internal::UNIT_STATIC(g_input_tex2)<typename I::value_type::cuda_bt>::tex());
    }

    dim3 dimgrid = grid_dimension(in1.domain(), dimblock);

    pw_call<meta_convolve_col_static_kernel2_sig(I::target, typename I::value_type::cuda_bt,
                                                 typename O::value_type, G1, G2, KERNEL_HALF_SIZE)>
      (flag<I::target>(), dimgrid, dimblock, in1, in2, out1, out2);

    // meta_convolve_internal::meta_convolve_col_static_kernel2
    //   <typename I::value_type::cuda_bt, typename O::value_type, G1, G2, KERNEL_HALF_SIZE>
    //   <<<dimgrid, dimblock, 0, stream>>>(mki(out1), mki(out2));

    if (I::target == GPU)
    {
      cudaUnbindTexture(meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());
      cudaUnbindTexture(meta_convolve_internal::UNIT_STATIC(g_input_tex2)<typename I::value_type::cuda_bt>::tex());
      check_cuda_error();
    }
  }

}

#endif
