#ifndef CUIMG_LARGE_MVT_DETECTOR_HPP_
# define  CUIMG_LARGE_MVT_DETECTOR_HPP_

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <host_defines.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>



#include <dige/window.h>
#include <dige/widgets/image_view.h>

#include <cuimg/dige.h>

# include <cuimg/dsl/binary_sub.h>
# include <cuimg/dsl/abs.h>

# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/fill.h>
# include <cuimg/gpu/mipmap.h>

# include <cuimg/gpu/kernel_util.h>
# include <cuimg/neighb2d_data.h>
# include <cuimg/neighb_iterator2d.h>
# include <cuimg/static_neighb2d.h>
# include <cuimg/point2d.h>


# include <cuimg/gpu/tracking/fast_tools.h>
# include <cuimg/gpu/tracking/fast_tools.h>

namespace cuimg
{

  template <typename V>
  large_mvt_detector<V>::large_mvt_detector(const domain_t& d)
    : display_(d),
      gl_frame_(d),
      particles_(domain_t(d.nrows() / 8, d.ncols() / 8)),
      particles2_(domain_t(d.nrows() / 8, d.ncols() / 8)),
      feature_(domain_t(d.nrows() / 8, d.ncols() / 8)),
      sa_(domain_t(d.nrows() / 8, d.ncols() / 8))
  {
    pyramid1_ = allocate_mipmap(i_float1(), display_, PS);
    pyramid2_ = allocate_mipmap(i_float1(), display_, PS);
    p1_ = &pyramid1_;
    p2_ = &pyramid2_;

    pyramid_tmp1_ = allocate_mipmap(i_float1(), display_, PS);
    pyramid_tmp2_ = allocate_mipmap(i_float1(), display_, PS);

    diff_pyramid_ = allocate_mipmap(i_float1(), display_, PS);
  }

  template <typename V>
  __global__ void relative_diff(const kernel_image2d<V> a,
                                const kernel_image2d<V> b,
                                kernel_image2d<V> out)
  {
    point2d<int> p = thread_pos2d();
    if (!a.has(p))
      return;

    float max_diff = 0.f;
    for_all_in_static_neighb2d(p, n, circle_r3) if (a.has(n))
    {
      float d = ::abs(a(p) - a(n));
      if (d > max_diff) max_diff = d;
    }

    if (max_diff > 0.f)
      out(p) = (a(p) - b(p)) / max_diff;
    else
      out(p) = 0.f;
  }

  template <typename P>
  __global__ void draw_pointsxxx(kernel_image2d<P> pts,
		       kernel_image2d<i_float4> out,
           int age_filter)
{
  point2d<int> p = thread_pos2d();
  if (!out.has(p))
    return;

  if (pts(p).age < age_filter) out(p) = i_float4(0.f, 0.f, 0.f, 1.f);
  else out(p) = i_float4(1.f, 0.f, 0, 1.f);
}


  template <typename P>
  __global__ void draw_pointsxxx(kernel_image2d<i_float1> frame,
		       kernel_image2d<P> pts,
		       kernel_image2d<i_float4> out,
           int age_filter)
{
  point2d<int> p = thread_pos2d();
  if (!frame.has(p))
    return;

  if (pts(p).age < age_filter) out(p) = i_float4(frame(p).x, frame(p).x, frame(p).x, 1.f);
  else out(p) = i_float4(1.f, 0, 0, 1.f);
}

  template <typename V>
  void
  large_mvt_detector<V>::estimate()
  {
    const int L = 3;
    host_image2d<particle_t>& ps;

    int rc_max = 0;
    int translation_max = 0;

    // Put every matches in a vector.
    for (unsigned r = 0; r < ps.nrows(); r++)
      for (unsigned c = 0; c < ps.ncols(); c++)
        if (matches(r, c).x != -1)
          mvts.push_back(mvt(point2d<int>(r, c), matches(r, c)));

    // Stats on translation
  }

  template <typename V>
  void
  large_mvt_detector<V>::update(const image2d<V>& in)
  {
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    std::swap(p1_, p2_);

    gl_frame_ = (get_x(in) + get_y(in) + get_z(in)) / 3.f;

    update_mipmap(gl_frame_, *p1_, pyramid_tmp1_, pyramid_tmp2_, PS);

    feature_.update((*p1_)[3]);
    sa_.update(feature_);

    for (unsigned i = 0; i < PS; i++)
      //        diff_pyramid_[i] = abs((*p1_)[i] - (*p2_)[i]);
      relative_diff<i_float1><<<dimgrid, dimblock>>>((*p1_)[i], (*p2_)[i], diff_pyramid_[i]);

#ifdef WITH_DISPLAY
  draw_pointsxxx<particle_t><<<dimgrid, dimblock>>>((*p1_)[3], sa_.particles(), particles_, Slider("age_filter").value());
  draw_pointsxxx<particle_t><<<dimgrid, dimblock>>>(sa_.particles(), particles2_, Slider("age_filter").value());
  ImageView("frame") <<= dg::dl() - particles2_ - particles_;

      dg::widgets::ImageView("mvt_detector")
        <<= dg::dl() - (*p1_)[0] - (*p1_)[1] - (*p1_)[2] - (*p1_)[3] - (*p1_)[4] +
            diff_pyramid_[0] - diff_pyramid_[1] - diff_pyramid_[2] - diff_pyramid_[3] - diff_pyramid_[4];
#endif

  }

}

#endif // !  CUIMG_LARGE_MVT_DETECTOR_HPP_
