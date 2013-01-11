#ifndef CUIMG_MIPMAP_H_
# define CUIMG_MIPMAP_H_

# include <cuimg/image_traits.h>
# include <cuimg/util.h>
# include <cuimg/pw_call.h>
# include <cuimg/gpu/mipmap.h>
# include <cuimg/copy.h>
# include <cuimg/gpu/texture.h>

# include <cuimg/meta_gaussian/meta_gaussian_coefs_1.h>
# include <cuimg/gpu/local_jet_static.h>

namespace cuimg
{
  namespace mipmap_internals
  {

    template <typename T>
    struct UNIT_STATIC(mipmap_input_tex);
      REGISTER_TEXTURE2D_PROXY(mipmap_input_tex);

    template <target T, typename I>
    __host__ __device__ void mipmap_kernel(thread_info<T> ti, kernel_image2d<I> in, kernel_image2d<I> out)
    {
      typedef typename I::cuda_bt V;
      i_int2 p = thread_pos2d(ti);

      if (!out.has(p))
        return;

      i_int2 d(p[0] * 2, p[1] * 2);
      out(p) = (I(tex2D(flag<T>(), UNIT_STATIC(mipmap_input_tex)<V>::tex(), in, d[1], d[0])) +
                I(tex2D(flag<T>(), UNIT_STATIC(mipmap_input_tex)<V>::tex(), in, d[1] + 1, d[0])) +
                I(tex2D(flag<T>(), UNIT_STATIC(mipmap_input_tex)<V>::tex(), in, d[1], d[0] + 1)) +
                I(tex2D(flag<T>(), UNIT_STATIC(mipmap_input_tex)<V>::tex(), in, d[1] + 1, d[0] + 1))) / 4.f;
    }

#define mipmap_kernel_sig(T, I) kernel_image2d<I>, kernel_image2d<I>, &mipmap_kernel<T, I>

  }

  template <typename D, typename S>
  std::vector<typename change_value_type<S, D>::ret>
  allocate_mipmap(const D&,
                  Image2d<S>& in,
                  unsigned nlevel,
                  dim3 dimblock = dim3(16, 16, 1))
  {
    using namespace mipmap_internals;

    typedef typename change_value_type<S, D>::ret DI;

    std::vector<DI> res(nlevel);

    res[0] = DI(exact(in).domain());

    DI c = res[0];
    for (unsigned l = 1; l < nlevel; l++)
    {
      DI gaussian(c.domain());
      DI tmp(c.domain());
      DI out(c.nrows() / 2, c.ncols() / 2);
      res[l] = out;
      c = out;
    }

    return res;
  }

  template <typename I>
  std::vector<I> build_mipmap(Image2d<I>& in,
                              unsigned nlevel,
                              dim3 dimblock = dim3(16, 16, 1))
  {
    using namespace mipmap_internals;

    typedef typename I::value_type U;
    typedef typename U::cuda_bt V;
    std::vector<I> res(nlevel);

    res[0] = exact(in);
    I c = exact(in);
    // for (unsigned l = 1; l < nlevel; l++)
    // {
    //   I gaussian(c.domain());
    //   I tmp(c.domain());
    //   I out(c.nrows() / 2, c.ncols() / 2);

    //   local_jet_static<I, I, I, 0, 0, 1, 6>(c, gaussian, tmp);

    //   dim3 dimgrid = grid_dimension(out.domain(), dimblock);
    //   bindTexture2d(gaussian, mipmap_internals::UNIT_STATIC(mipmap_input_tex)<V>::tex());
    //   mipmap_kernel<<<dimgrid, dimblock>>>(mki(out));
    //   cudaUnbindTexture(mipmap_internals::UNIT_STATIC(mipmap_input_tex)<V>::tex());
    //   res[l] = out;
    //   c = out;
    // }

    return res;
  }

  /*
  template <typename U>
  void build_fast_mipmap(device_image2d<U>& in,
                         device_image2d<U>& out,
                         unsigned nlevel,
                         dim3 dimblock = (16, 16, 1))
  {
    using namespace mipmap_internals;

    typedef typename U::cuda_bt V;
    std::vector<device_image2d<U> > res(nlevel);

    res[0] = in;
    device_image2d<U> c = in;
    for (unsigned l = 1; l < nlevel; l++)
    {
      device_image2d<U> gaussian(c.domain());
      device_image2d<U> tmp(c.domain());
      device_image2d<U> out(c.nrows() / 2, c.ncols() / 2);

      local_jet_static<device_image2d<U>, device_image2d<U>, device_image2d<U>, 0, 0, 1, 6>(c, gaussian, tmp);

      dim3 dimgrid = grid_dimension(out.domain(), dimblock);
      bindTexture2d(gaussian, mipmap_internals::UNIT_STATIC(mipmap_input_tex)<V>::tex());
      mipmap_kernel<<<dimgrid, dimblock>>>(mki(out));
      cudaUnbindTexture(mipmap_internals::UNIT_STATIC(mipmap_input_tex)<V>::tex());
      res[l] = out;
      c = out;
    }

    return res;
  }
  */

  template <typename I>
  void update_mipmap(const Image2d<I>& in,
                     std::vector<I>& pyramid_out,
                     std::vector<I>& pyramid_tmp1,
                     std::vector<I>& pyramid_tmp2,
                     unsigned nlevel,
                     cudaStream_t stream = 0,
                     dim3 dimblock = dim3(16, 16, 1))
  {
    SCOPE_PROF(update_mipmap);
    using namespace mipmap_internals;

    typedef typename I::value_type U;
    typedef typename U::cuda_bt V;

    START_PROF(copy);
    copy(exact(in), pyramid_out[0]);
    END_PROF(copy);
    /* local_jet_static<I, I, I, 0, 0, 1, 1> */
    /*   (exact(in), pyramid_out[0], pyramid_tmp1[0], stream, dimblock); */

    I c = pyramid_out[0];

    for (unsigned l = 1; l < nlevel; l++)
    {
      I gaussian = pyramid_tmp1[l-1];
      I& tmp = pyramid_tmp2[l-1];
      I& out = pyramid_out[l];

      /* local_jet_static<I, I, I, 0, 0, 1, 1> */
      /*   (c, gaussian, tmp, stream, dimblock); */
      /* START_PROF(copy); */
      /* copy(c, gaussian); */
      /* END_PROF(copy); */

      dim3 dimgrid = grid_dimension(out.domain(), dimblock);

#ifndef NO_CUDA
      if (I::target == GPU)
        bindTexture2d(gaussian, mipmap_internals::UNIT_STATIC(mipmap_input_tex)<V>::tex());
#endif

      START_PROF(resize_kernel);
      pw_call<mipmap_kernel_sig(I::target, U)>(flag<I::target>(), dimgrid, dimblock, c, out);
      END_PROF(resize_kernel);

#ifndef NO_CUDA
      if (I::target == GPU)
        cudaUnbindTexture(mipmap_internals::UNIT_STATIC(mipmap_input_tex)<V>::tex());
#endif

      c = out;

      if (I::target == GPU)
        check_cuda_error();
    }

  }

  template <typename I>
  void subsample(const Image2d<I>& in,
		 Image2d<I>& out)
  {
    SCOPE_PROF(subsample);
    dim3 dimblock = dim3(exact(out).domain().ncols(), 1, 1);
    using namespace mipmap_internals;
    dim3 dimgrid = grid_dimension(exact(out).domain(), dimblock);

    /* I tmp2(exact(in).domain()); */
    /* cudaStream_t stream = 0; */
    /* local_jet_static<I, I, I, 0, 0, 2, 2> */
    /*   (exact(in), exact(tmp2), exact(tmp), stream, dimblock); */

    pw_call<mipmap_kernel_sig(I::target, typename I::value_type)>(flag<I::target>(), dimgrid, dimblock,
    								  exact(in), exact(out));
  }

  template <typename I>
  void update_mipmap_level(const Image2d<I>& in,
                           std::vector<I>& pyramid_out,
                           unsigned level,
                           cudaStream_t stream = 0,
                           dim3 dimblock = dim3(16, 16, 1))
  {
    SCOPE_PROF(update_mipmap_level);
    using namespace mipmap_internals;

    typedef typename I::value_type U;
    typedef typename U::cuda_bt V;

    if (level == 0)
    {
      START_PROF(copy);
      copy(exact(in), pyramid_out[0]);
      END_PROF(copy);
    }
    else
    {
      dim3 dimgrid = grid_dimension(pyramid_out[level].domain(), dimblock);
      START_PROF(resize_kernel);
      pw_call<mipmap_kernel_sig(I::target, U)>(flag<I::target>(), dimgrid, dimblock,
                                               pyramid_out[level-1], pyramid_out[level]);
      END_PROF(resize_kernel);
    }

  }

}

#endif
