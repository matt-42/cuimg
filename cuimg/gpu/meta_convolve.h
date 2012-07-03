#ifndef CUIMG_META_CONVOLVE_STATIC_H_
# define CUIMG_META_CONVOLVE_STATIC_H_

# include <cuimg/util.h>
# include <cuimg/error.h>
# include <cuimg/gpu/texture.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/gpu/device_image2d.h>
# include <cuimg/gaussian_kernel.h>
# include <cuimg/meta_gaussian/meta_gaussian.h>

namespace cuimg
{
  namespace meta_convolve_internal
  {

    template <typename T>
    struct UNIT_STATIC(g_input_tex);
    REGISTER_TEXTURE2D_PROXY(g_input_tex);

    template <typename T>
    struct UNIT_STATIC(g_input_tex2);
    REGISTER_TEXTURE2D_PROXY(g_input_tex2);

    template <typename I, int R, int E, typename G>
      struct meta_convolve2d_row_loop
      {
        template <unsigned T, typename U>
        static __host__ __device__ inline U iter(const kernel_image2d<U>& in, const i_int2& p)
        {;
          if (T == GPU)
            return U(tex2D(UNIT_STATIC(g_input_tex)<typename U::cuda_bt>::tex(), p.y + R, p.x)) * meta_kernel<G, R>::coef() +
            meta_convolve2d_row_loop<I, R + 1, E, G>::template iter<T, U>(in, p);
          else if (T == CPU)
            return U(in(p.x, p.y + R)) * meta_kernel<G, R>::coef() +
            meta_convolve2d_row_loop<I, R + 1, E, G>::template iter<T, U>(in, p);

        }
      };

    template <typename I, int E, typename G>
      struct meta_convolve2d_row_loop<I, E, E, G>
    {
      template <unsigned T, typename U>
        static __host__ __device__ inline U iter(const kernel_image2d<U>& in,
                                        const i_int2& p)
      {
        if (T == GPU)
          return U(tex2D(UNIT_STATIC(g_input_tex)<I>::tex(), p.y + E, p.x)) * meta_kernel<G, E>::coef();
        else
          return in(p.x, p.y + E) * meta_kernel<G, E>::coef();
      }
    };

#define meta_convolve_row_static_kernel_sig(T, I, O, G, KERNEL_HALF_SIZE) \
    kernel_image2d<O>, kernel_image2d<O>,                               \
    &meta_convolve_internal::meta_convolve_row_static_kernel<T, I, O, G, KERNEL_HALF_SIZE>

    template <target T, typename I, typename O, typename G, int KERNEL_HALF_SIZE>
    __host__ __device__ void meta_convolve_row_static_kernel(thread_info<T> ti,
                                                             kernel_image2d<O> in,
                                                             kernel_image2d<O> out)
    {
      int idr = ti.blockIdx.y * ti.blockDim.y + ti.threadIdx.y;
      int idc = ti.blockIdx.x * ti.blockDim.x + ti.threadIdx.x;
      i_int2 p(idr, idc);
      if (!out.has(p))
        return;
      if (T == CPU && (p.x < KERNEL_HALF_SIZE || p.y < KERNEL_HALF_SIZE ||
                       p.x >= in.nrows() - KERNEL_HALF_SIZE || p.y >= in.ncols() - KERNEL_HALF_SIZE))
        return;
      out(p) = meta_convolve2d_row_loop<I, -KERNEL_HALF_SIZE, KERNEL_HALF_SIZE, G>::template iter<T, O>(in, p);
    }

    template <typename I, int R, int E, typename G>
      struct meta_convolve2d_col_loop
      {
        template <unsigned T, typename U>
        static __host__ __device__ inline U iter(const kernel_image2d<U>& in,
                                        const kernel_image2d<U>& out, const i_int2& p)
        {
          if (T == GPU)
            return U(tex2D(UNIT_STATIC(g_input_tex)<typename U::cuda_bt>::tex(),
                           p.y, p.x + R)) * meta_kernel<G, R>::coef() +
              meta_convolve2d_col_loop<I, R + 1, E, G>::template iter<T, U>(in, out, p);
          else
            return in(p.x + R, p.y) * meta_kernel<G, R>::coef() +
              meta_convolve2d_col_loop<I, R + 1, E, G>::template iter<T, U>(in, out, p);
        }
      };

    template <typename I, int E, typename G>
      struct meta_convolve2d_col_loop<I, E, E, G>
    {
      template <unsigned T, typename U>
        static __host__ __device__ inline U iter(const kernel_image2d<U>& in,
                                        const kernel_image2d<U>& out,
                                        const i_int2& p)
      {
        if (T == GPU)
          return U(tex2D(UNIT_STATIC(g_input_tex)<I>::tex(),
                         p.y, p.x + E)) * meta_kernel<G, E>::coef();
        else
          return U(in(p.x + E, p.y)) * meta_kernel<G, E>::coef();

      }
    };


#define meta_convolve_col_static_kernel_sig(T, I, O, G, KERNEL_HALF_SIZE) \
    kernel_image2d<O>, kernel_image2d<O>,                               \
    &meta_convolve_internal::meta_convolve_col_static_kernel<T, I, O, G, KERNEL_HALF_SIZE>

    template <target T, typename I, typename O, typename G, int KERNEL_HALF_SIZE>
    __host__ __device__ void meta_convolve_col_static_kernel(thread_info<T> ti,
                                                             kernel_image2d<O> in,
                                                             kernel_image2d<O> out)
    {
      int idr = ti.blockIdx.y * ti.blockDim.y + ti.threadIdx.y;
      int idc = ti.blockIdx.x * ti.blockDim.x + ti.threadIdx.x;
      i_int2 p(idr, idc);
      if (!out.has(p))
        return;
      if (T == CPU && (p.x < KERNEL_HALF_SIZE || p.y < KERNEL_HALF_SIZE ||
                       p.x >= in.nrows() - KERNEL_HALF_SIZE || p.y >= in.ncols() - KERNEL_HALF_SIZE))
        return;
      out(p) = meta_convolve2d_col_loop<I, -KERNEL_HALF_SIZE, KERNEL_HALF_SIZE, G>::template iter<T, O>(in, out, p);
    }
  }

  template <typename I, typename O, typename G, int KERNEL_HALF_SIZE>
  void meta_convolve_row2d(const I& in, O& out,
                           cudaStream_t stream = 0, dim3 dimblock = dim3(16, 16))
  {
    assert(in.domain() == out.domain());

    /* if (I::target == CPU) */
    /*   return meta_convolve_row2d_cpu<I::value_type, O::value_type, (in, out); */

    if (I::target == GPU)
      bindTexture2d(in, meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());

    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    pw_call<meta_convolve_row_static_kernel_sig(I::target, typename I::value_type::cuda_bt,
                                                typename O::value_type, G, KERNEL_HALF_SIZE)>
      (flag<I::target>(), dimgrid, dimblock, in, out);

    // meta_convolve_internal::meta_convolve_row_static_kernel
    // <typename I::value_type::cuda_bt, typename O::value_type, G, KERNEL_HALF_SIZE>
    //   <<<dimgrid, dimblock, 0, stream>>>(mki(out));

#ifndef NO_CUDA
    if (I::target == GPU)
    {
      cudaUnbindTexture(meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());
      check_cuda_error();
    }
#endif

  }

//   template <typename I, typename O, typename G, int KERNEL_HALF_SIZE>
//   void meta_convolve_row2d(const host_image2d<typename I::value_type>& in,
// 			       host_image2d<typename O::value_type>& out,
// 			       cudaStream_t stream, dim3 dimblock)
//   {
//     typedef typename I::value_type V;
//     assert(in.domain() == out.domain());

// #pragma omp parallel for schedule(static, 8)
//     for (unsigned r = KERNEL_HALF_SIZE; r < in.nrows() - KERNEL_HALF_SIZE; r++)
//       for (unsigned c = KERNEL_HALF_SIZE; c < in.ncols() - KERNEL_HALF_SIZE; c++)
//       {
// 	i_int2 p(r, c);
// 	out(p) = meta_convolve_internal::meta_convolve2d_row_loop
// 	  <typename I::value_type::cuda_bt, -KERNEL_HALF_SIZE, KERNEL_HALF_SIZE, G>
// 	  ::template iter<CPU, typename I::value_type>(mki(in), p);
//       }

//   }

  template <typename I, typename O, typename G, int KERNEL_HALF_SIZE>
  void meta_convolve_row2d(const host_image2d<typename I::value_type>& in,
			       host_image2d<typename O::value_type>& out,
			       cudaStream_t stream, dim3 dimblock)
  {
    typedef typename I::value_type V;
    typedef typename change_coord_type<float, V>::ret VI;
    assert(in.domain() == out.domain());

    float kernel[KERNEL_HALF_SIZE*2];
    for (unsigned i = 0; i < KERNEL_HALF_SIZE * 2; i++)
      kernel[i] = gaussian_derivative<0>(G::s, i - KERNEL_HALF_SIZE);

#pragma omp parallel for schedule(static, 2)
    for (unsigned r = KERNEL_HALF_SIZE; r < in.nrows() - KERNEL_HALF_SIZE; r++)
      for (unsigned c = KERNEL_HALF_SIZE; c < in.ncols() - KERNEL_HALF_SIZE; c++)
      {
	VI res = zero();
	for (unsigned k = 0; k < KERNEL_HALF_SIZE * 2; k++)
	  res += in(r, c + k - KERNEL_HALF_SIZE) * kernel[k];
	out(r, c) = res;
      }

  }

  template <typename I, typename O, typename G, int KERNEL_HALF_SIZE>
  void meta_convolve_col2d(const host_image2d<typename I::value_type>& in,
			   host_image2d<typename O::value_type>& out,
			   cudaStream_t stream, dim3 dimblock)
  {
    typedef typename I::value_type V;
    typedef typename change_coord_type<float, V>::ret VI;
    assert(in.domain() == out.domain());

    float kernel[KERNEL_HALF_SIZE*2+1];
    for (unsigned i = 0; i < KERNEL_HALF_SIZE * 2 + 1; i++)
      kernel[i] = gaussian_derivative<0>(G::s, i - KERNEL_HALF_SIZE);

#pragma omp parallel for schedule(static, 2)
    for (unsigned r = KERNEL_HALF_SIZE; r < in.nrows() - KERNEL_HALF_SIZE; r++)
      for (unsigned c = KERNEL_HALF_SIZE; c < in.ncols() - KERNEL_HALF_SIZE; c++)
      {
	VI res = zero();
	for (unsigned k = 0; k < KERNEL_HALF_SIZE * 2 + 1; k++)
	  res += in(r + k - KERNEL_HALF_SIZE, c) * kernel[k];
	out(r, c) = res;
      }

  }


// #define WS 8

//   template <typename I, typename O, typename G, int KERNEL_HALF_SIZE>
//   void meta_convolve_row2d(const host_image2d<typename I::value_type>& in,
// 			       host_image2d<typename O::value_type>& out,
// 			       cudaStream_t stream, dim3 dimblock)
//   {
//     typedef typename I::value_type V;
//     assert(in.domain() == out.domain());

//     float kernel[KERNEL_HALF_SIZE*2+1];
//     for (unsigned i = 0; i < KERNEL_HALF_SIZE * 2 + 1; i++)
//       kernel[i] = gaussian_derivative<0>(G::s, i - KERNEL_HALF_SIZE);

//     const int ks = KERNEL_HALF_SIZE * 2 + 1;
//     const int hks = KERNEL_HALF_SIZE;
//     const int hks2 = KERNEL_HALF_SIZE * 2;
// #pragma omp parallel for schedule(static, 4)
//     for (unsigned r = 0; r < in.nrows(); r++)
//     {
//       V sums[WS];
//       V vs[hks2+WS];

//       const V* inrow = in.row(r);
//       V* outrow = out.row(r);

//       for (unsigned k = hks2; k < hks2+WS; k++)
//         vs[k] = inrow[k-hks2];

//       for (unsigned c = hks; c < in.ncols()-hks-WS; c+=WS)
//       {
//         unsigned off = c;
//         for (unsigned l = 0; l < WS; l++)
//           sums[l] = 0.f;

//         for (unsigned k = 0; k < hks2; k++)
//           vs[k] = vs[WS+k];
//         for (unsigned k = hks2; k < hks2+WS; k++)
//           vs[k] = inrow[off+k-hks];

//         for (unsigned k = 0; k < ks; k++)
//           for (unsigned l = 0; l < WS; l++)
//           {
//             sums[l] += vs[k+l] * kernel[k];
//           }

//         for (unsigned l = 0; l < WS; l++)
//           outrow[off+l] = sums[l];
//       }

//     }

//   }

  template <typename I, typename O, typename G, int KERNEL_HALF_SIZE>
  void meta_convolve_col2d(const I& in, O& out,
                           cudaStream_t stream = 0, dim3 dimblock = dim3(16, 16))
  {
    assert(in.domain() == out.domain());

    if (I::target == GPU)
      bindTexture2d(in, meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());

    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    pw_call<meta_convolve_col_static_kernel_sig(I::target, typename I::value_type::cuda_bt,
                                                typename O::value_type, G, KERNEL_HALF_SIZE)>
      (flag<I::target>(), dimgrid, dimblock, in, out);

    // meta_convolve_internal::meta_convolve_col_static_kernel
    // <typename I::value_type::cuda_bt, typename O::value_type, G, KERNEL_HALF_SIZE>
    //   <<<dimgrid, dimblock, 0, stream>>>(mki(out));

#ifndef NO_CUDA
    if (I::target == GPU)
    {
      cudaUnbindTexture(meta_convolve_internal::UNIT_STATIC(g_input_tex)<typename I::value_type::cuda_bt>::tex());
      check_cuda_error();
    }
#endif
  }

}

#endif
