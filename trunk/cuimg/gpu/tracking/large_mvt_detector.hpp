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


# include <cuimg/cpu/host_image2d.h>
# include <cuimg/cpu/fill.h>

# include <cuimg/neighb2d_data.h>
# include <cuimg/neighb_iterator2d.h>
# include <cuimg/static_neighb2d.h>
# include <cuimg/point2d.h>


# include <cuimg/gpu/tracking/fast_tools.h>
# include <cuimg/gpu/tracking/fast_tools.h>

namespace cuimg
{

  template <typename V>
  large_mvt_detector<V>::large_mvt_detector()
    : h(400, 400)
  {
    mvts.reserve(400);
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

  template <typename V>
  i_short2
  large_mvt_detector<V>::estimate(const host_image2d<i_short2>& matches)
  {
    mvts.clear();
    // Put every matches in a vector.
    for (unsigned r = 0; r < matches.nrows(); r++)
      for (unsigned c = 0; c < matches.ncols(); c++)
        if (matches(r, c).x > 0)
          mvts.push_back(mvt(point2d<int>(r, c), matches(r, c) - i_int2(r, c)));

    i_int2 h_center(h.nrows() / 2, h.ncols() / 2);
    fill(h, unsigned short(0));

    // Stats on translation
    tr_max_ = i_char2(0, 0);
    tr_max_cpt_ = 0;
    for (unsigned i = 0; i < mvts.size(); i++)
    {
      if (::abs(mvts[i].tr.x) >= h.nrows() / 2 || ::abs(mvts[i].tr.y) >= h.ncols() / 2) continue;
      int c = ++h(mvts[i].tr + h_center);
      if (c > tr_max_cpt_)
      {
        tr_max_cpt_ = c;
        tr_max_ = mvts[i].tr;
      }
    }

    //if (tr_max_cpt_ / float(mvts.size()) > 0.10f)
      return tr_max_;
    //else
    //  return i_short2(0,0);
  }

  template <typename V>
  void
  large_mvt_detector<V>::display()
  {
#ifdef WITH_DISPLAY
    if (tr_max_cpt_)
      for (unsigned r = 0; r < h.nrows(); r++)
        for (unsigned c = 0; c < h.ncols(); c++)
          h(r, c) = (65535 * int(h(r, c))) / (tr_max_cpt_);

    ImageView("big_mvt") <<= dg::dl() - h;
    // Test.
    if (tr_max_cpt_ > (mvts.size() / 20) &&
        tr_max_ != i_char2(0, 0))
    {
      std::cout << "Big translation detected: " << 100*tr_max_cpt_/mvts.size() << "%\t of "
                << mvts.size() << " matches: " << i_int2(tr_max_) << std::endl;
    }
#endif
  }

}

#endif // !  CUIMG_LARGE_MVT_DETECTOR_HPP_
