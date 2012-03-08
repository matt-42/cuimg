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

# include <cuimg/gpu/device_image2d.h>
# include <cuimg/gpu/fill.h>
# include <cuimg/gpu/mipmap.h>
# include <cuimg/gpu/kernel_util.h>


# include <cuimg/cpu/host_image2d.h>
# include <cuimg/cpu/fill.h>

# include <cuimg/neighb2d_data.h>
# include <cuimg/neighb_iterator2d.h>
# include <cuimg/static_neighb2d.h>
# include <cuimg/point2d.h>


# include <cuimg/tracking/fast_tools.h>

# include <dige/widgets/image_view.h>


namespace cuimg
{

  template <typename V>
  large_mvt_detector<V>::large_mvt_detector()
    : h(400, 400)
  {
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
  template <typename P>
  i_short2
  large_mvt_detector<V>::estimate(const thrust::host_vector<P>& particles,
                                  unsigned n_particles)
  {
    n_particles_ = n_particles;
    typedef unsigned short US;
    tr_max_ = i_char2(0, 0);
    tr_max_cpt_ = 0;

    if (n_particles < 20) return tr_max_;

    i_int2 h_center(h.nrows() / 2, h.ncols() / 2);
    fill(h, US(0));

    // Stats on translation
    for (unsigned i = 0; i < n_particles; i++)
    {
      if (particles[i].age < 5 ||
          ::abs(particles[i].brut_acceleration.x) >= h.nrows() / 2 ||
          ::abs(particles[i].brut_acceleration.y) >= h.ncols() / 2) continue;
      int c = ++h(particles[i].brut_acceleration + h_center);
      if (c > tr_max_cpt_)
      {
        tr_max_cpt_ = c;
        tr_max_ = particles[i].brut_acceleration;
      }
    }

    return tr_max_;
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

    dg::widgets::ImageView("big_mvt") <<= dg::dl() - h;
    // Test.
    if (tr_max_ != i_char2(0, 0))
    {
      std::cout << "Big translation detected: " << 100*tr_max_cpt_/n_particles_ << "%\t of "
                << n_particles_ << " matches: " << i_int2(tr_max_) << std::endl;
    }
#endif
  }

}

#endif // !  CUIMG_LARGE_MVT_DETECTOR_HPP_
