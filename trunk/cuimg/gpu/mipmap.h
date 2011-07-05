#ifndef CUIMG_MIPMAP_H_
# define CUIMG_MIPMAP_H_

# include <cuimg/util.h>
# include <cuimg/gpu/mipmap.h>
# include <cuimg/gpu/texture.h>

# include <cuimg/meta_gaussian/meta_gaussian_coef_1.h>
# include <cuimg/gpu/local_jet_static.h>

namespace cuimg
{
  namespace mipmap_internals
  {

    template <typename T>
    struct UNIT_STATIC(mipmap_input_tex);
      REGISTER_TEXTURE2D_PROXY(mipmap_input_tex);

    template <typename I>
    __global__ void mipmap_kernel(kernel_image2d<I> out)
    {
      typedef typename I::cuda_bt V;
      i_int2 p = thread_pos2d();

      if (!out.has(p))
        return;

      i_int2 d(p[0] * 2, p[1] * 2);
      out(p) = (I(tex2D(UNIT_STATIC(mipmap_input_tex)<V>::tex(), d[1], d[0])) +
                I(tex2D(UNIT_STATIC(mipmap_input_tex)<V>::tex(), d[1] + 1, d[0])) +
                I(tex2D(UNIT_STATIC(mipmap_input_tex)<V>::tex(), d[1], d[0] + 1)) +
                I(tex2D(UNIT_STATIC(mipmap_input_tex)<V>::tex(), d[1] + 1, d[0] + 1))) / 4.f;
    }

  }

  template <typename U>
  std::vector<image2d<U> > build_mipmap(image2d<U>& in,
                                        unsigned nlevel,
                                        dim3 dimblock = (16, 16, 1))
  {
    using namespace mipmap_internals;

    typedef typename U::cuda_bt V;
    std::vector<image2d<U> > res(nlevel);

    res[0] = in;
    image2d<U> c = in;
    for (unsigned l = 1; l < nlevel; l++)
    {
      image2d<U> gaussian(c.domain());
      image2d<U> tmp(c.domain());
      image2d<U> out(c.nrows() / 2, c.ncols() / 2);

      local_jet_static<image2d<U>, image2d<U>, image2d<U>, 0, 0, 1, 6>(c, gaussian, tmp);

      dim3 dimgrid = grid_dimension(out.domain(), dimblock);
      bindTexture2d(gaussian, mipmap_internals::UNIT_STATIC(mipmap_input_tex)<V>::tex());
      mipmap_kernel<<<dimgrid, dimblock>>>(mki(out));
      cudaUnbindTexture(mipmap_internals::UNIT_STATIC(mipmap_input_tex)<V>::tex());
      res[l] = out;
      c = out;
    }

    return res;
  }

}

#endif
