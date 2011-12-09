#ifndef CUIMG_FAST382S_FEATURE_HPP_
# define CUIMG_FAST382S_FEATURE_HPP_

# include <cuda_runtime.h>
# include <cuimg/gpu/local_jet_static.h>
# include <cuimg/dsl/binary_div.h>
# include <cuimg/dsl/binary_add.h>
# include <cuimg/dsl/get_comp.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_1.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_2.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_4.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_3.h>
# include <cuimg/gpu/tracking/fast_tools.h>

# include <cuimg/dige.h>

#include <dige/widgets/image_view.h>

using dg::dl;
using namespace dg::widgets;

namespace cuimg
{


  inline
  __host__ __device__
  float distance_mean(const dfast382s& a, const dfast382s& b)
  {
    if (::abs(a.pertinence - b.pertinence) > 0.1f ||
        b.pertinence < 0.15f ||
        a.pertinence < 0.15f
        )
      return 99999.f;
    else
    {
      float d = 0.f;
      for (unsigned i = 0; i < 16; i++)
      {
        float tmp = a.distances[i] - b.distances[i];
        d += tmp * tmp;
      }

      return ::sqrt(d) / ::sqrt(16.f);
    }
  }


  inline
  __host__ __device__
  float distance_mean_s2(const dfast382s& a, const dfast382s& b)
  {
    // if (::abs(a.pertinence - b.pertinence) > 0.1f ||
    //     b.pertinence < 0.15f ||
    //     a.pertinence < 0.15f)
    //   return 99999.f;
    // else
    {
      float d = 0.f;
      for (unsigned i = 8; i < 16; i++)
      {
        float tmp = a.distances[i] - b.distances[i];
        d += tmp * tmp;
      }

      return ::sqrt(d) / ::sqrt(8.f);
    }
  }


  inline
  __host__ __device__
  float distance_min(const dfast382s& a, const dfast382s& b)
  {
    if (::abs(a.pertinence - b.pertinence) > 0.1f ||
        b.pertinence < 0.15f ||
        a.pertinence < 0.15f)
      return 99999.f;
    else
    {
      float d = 9999999.f;
      for (unsigned i = 0; i < 16; i++)
      {
        float tmp = ::abs(a.distances[i] - b.distances[i]);
        if (tmp < d)
          d = tmp;
      }

      return d;
    }
  }

  inline
  __host__ __device__
  float distance_max(const dfast382s& a, const dfast382s& b)
  {
    if (::abs(a.pertinence - b.pertinence) > 0.1f ||
        b.pertinence < 0.15f ||
        a.pertinence < 0.15f)
      return 99999.f;
    else
    {
      float d = 0.f;
      for (unsigned i = 0; i < 16; i++)
      {
        float tmp = ::abs(a.distances[i] - b.distances[i]);
        if (tmp > d)
          d = tmp;
      }

      return d;
    }
  }

  __host__ __device__ inline
  dfast382s operator+(const dfast382s& a, const dfast382s& b)
  {
    dfast382s res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] + b.distances[i];
    res.pertinence = a.pertinence + b.pertinence;
    return res;
  }

  __host__ __device__ inline
  dfast382s operator-(const dfast382s& a, const dfast382s& b)
  {
    dfast382s res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] - b.distances[i];
    res.pertinence = a.pertinence - b.pertinence;
    return res;
  }

  template <typename S>
  __host__ __device__ inline
  dfast382s operator/(const dfast382s& a, const S& s)
  {
    dfast382s res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] / s;
    res.pertinence = a.pertinence / s;
    return res;
  }

  template <typename S>
  __host__ __device__ inline
  dfast382s operator*(const dfast382s& a, const S& s)
  {
    dfast382s res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] * s;
    res.pertinence = a.pertinence * s;
    return res;
  }


   __constant__ const int circle_r3_fast38s1[8][2] = {
     {-3, 0}, {-2, 2},
     { 0, 3}, { 2, 2},
     { 3, 0}, { 2,-2},
     { 0,-3}, {-2,-2}
   };
   __constant__ const int circle_r3_fast38s2[8][2] = {
     {-9, 0}, {-6, 6},
     { 0, 9}, { 6, 6},
     { 9, 0}, { 6,-6},
     { 0,-9}, {-6,-6}
   };
  /*
  __constant__ const int circle_r3_fast382s[8][2] = {
    {-1, 0}, {-1, 1},
    { 0, 1}, { 1, 1},
    { 1, 0}, { 1,-1},
    { 0,-1}, {-1,-1}
  };
  */
  template <typename V>
  __global__ void FAST382S(kernel_image2d<i_float4> frame_color,
                       kernel_image2d<V> frame_s1,
                       kernel_image2d<V> frame_s2,
                       kernel_image2d<dfast382s> out,
                       kernel_image2d<i_float1> pertinence,
                       float grad_thresh)
  {
    point2d<int> p = thread_pos2d();
    if (!frame_s1.has(p))//; || track(p).x == 0)
      return;

    float distances[16];
    float plj = frame_s1(p).x;
    {
      for(unsigned i = 0; i < 8; i++)
      {
        point2d<int> n1(p.row() + circle_r3_fast38s1[i][0],
                        p.col() + circle_r3_fast38s1[i][1]);
        if (frame_s1.has(n1))
          distances[i] = frame_s1(n1) - frame_s1(p);
        else
          distances[i] = 0.f;
      }

      for(unsigned i = 0; i < 8; i++)
      {
        point2d<int> n1(p.row() + circle_r3_fast38s2[i][0],
                        p.col() + circle_r3_fast38s2[i][1]);
        if (frame_s1.has(n1))
          distances[i+8] = frame_s2(n1) - frame_s2(p);
        else
          distances[i+8] = 0.f;
      }
    }

    {
      float min_diff = 9999999.f;
      float max_diff = 0.f;
      float max_single_diff = 0.f;
      unsigned min_orientation = 0;
      bool sign = false;
      for(unsigned i = 0; i < 8; i++)
      {
        point2d<int> n1(p.row() + circle_r3[i][0],
                        p.col() + circle_r3[i][1]);
        point2d<int> n2(p.row() + circle_r3[(i+8)][0],
                        p.col() + circle_r3[(i+8)][1]);

        if (frame_s1.has(n1) && frame_s1.has(n2))
        {
          float diff = plj - (frame_s1(n1).x + frame_s1(n2).x) / 2.f;
          float adiff = ::abs(diff);
          if (adiff < min_diff)
          {
            min_diff = adiff;
            min_orientation = i;
            sign = diff > 0;
          }
          float sd = ::max(::abs(plj - frame_s1(n1).x), ::abs(plj - frame_s1(n2).x));
          if (max_single_diff < sd) max_single_diff = sd;

        }
      }

      for(unsigned i = 0; i < 16; i++)
        out(p).distances[i] = distances[i] / max_single_diff;
      if (max_single_diff >= grad_thresh)
      {
        out(p).pertinence = min_diff / max_single_diff;
        pertinence(p) = min_diff / max_single_diff;//min(min_diff / max_single_diff, max_single_diff);

        //out(p).pertinence = 1.f;
        //pertinence(p) = 1.f;

      }
      else
      {
        pertinence(p) = 0.f;
        out(p).pertinence = 0.f;
      }

    }

  }

  template <typename T>
  __global__  void dfast382s_to_color(kernel_image2d<dfast382s> in,
                                  kernel_image2d<i_float4> out)
  {
    point2d<int> p = thread_pos2d();
    if (!in.has(p))
      return;

    i_float4 res;
    for (unsigned i = 0; i < 8; i++)
      res[i/2] = in(p).distances[i] / 2.f;
    res.w = 1.f;
    out(p) = res;
  }

  inline
  fast382s_feature::fast382s_feature(const domain_t& d)
    : gl_frame_(d),
      blurred_s1_(d),
      blurred_s2_(d),
      tmp_(d),
      pertinence_(d),
      f1_(d),
      f2_(d),
      fast382s_color_(d),
      color_blurred_(d),
      color_tmp_(d),
      grad_thresh(0.3f)
  {
    f_prev_ = &f1_;
    f_ = &f2_;
  }

  inline
  void
  fast382s_feature::update(const image2d<i_float4>& in)
  {
    gl_frame_ = (get_x(in) + get_y(in) + get_z(in)) / 3.f;
    swap_buffers();
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    local_jet_static_<0, 0, 2, 10>::run(gl_frame_, blurred_s1_, tmp_);
    local_jet_static_<0, 0, 4, 15>::run(gl_frame_, blurred_s2_, tmp_);
    //local_jet_static_<0, 0, 1, 5>::run(in, color_blurred_, color_tmp_);


    grad_thresh = Slider("grad_thresh").value() / 100.f;
    FAST382S<i_float1><<<dimgrid, dimblock>>>
      (color_blurred_, blurred_s1_, blurred_s2_, *f_, pertinence_, grad_thresh);

#ifdef WITH_DISPLAY
    dfast382s_to_color<int><<<dimgrid, dimblock>>>(*f_, fast382s_color_);
    ImageView("test") <<= dg::dl() - gl_frame_ - pertinence_ - fast382s_color_;
#endif

    check_cuda_error();
  }

  inline
  const image2d<i_float4>&
  fast382s_feature::feature_color() const
  {
    return fast382s_color_;
  }

  inline
  void
  fast382s_feature::swap_buffers()
  {
    std::swap(f_prev_, f_);
  }

  inline
  const fast382s_feature::domain_t&
  fast382s_feature::domain() const
  {
    return f1_.domain();
  }

  inline
  image2d<dfast382s>&
  fast382s_feature::previous_frame()
  {
    return *f_prev_;
  }

  inline
  image2d<dfast382s>&
  fast382s_feature::current_frame()
  {
    return *f_;
  }

  inline
  image2d<i_float1>&
  fast382s_feature::pertinence()
  {
    return pertinence_;
  }

  inline
  kernel_fast382s_feature::kernel_fast382s_feature(fast382s_feature& f)
    : pertinence_(f.pertinence()),
      f_prev_(f.previous_frame()),
      f_(f.current_frame())
  {
  }

  inline
  __device__ float
  kernel_fast382s_feature::distance(const point2d<int>& p_prev,
                                const point2d<int>& p_cur)
  {
    return cuimg::distance_mean(f_prev_(p_prev), f_(p_cur));
  }

  inline
  __device__ float
  kernel_fast382s_feature::distance(const dfast382s& a,
                                const dfast382s& b)
  {
    return cuimg::distance_mean(a, b);
  }

  inline
  __device__ float
  kernel_fast382s_feature::distance_s2(const dfast382s& a,
                                const dfast382s& b)
  {
    return cuimg::distance_mean_s2(a, b);
  }

  inline __device__
  kernel_image2d<dfast382s>&
  kernel_fast382s_feature::previous_frame()
  {
    return f_prev_;
  }

  inline __device__
  kernel_image2d<dfast382s>&
  kernel_fast382s_feature::current_frame()
  {
    return f_;
  }

  inline __device__
  kernel_image2d<i_float1>&
  kernel_fast382s_feature::pertinence()
  {
    return pertinence_;
  }

}

#endif // ! CUIMG_FAST382S_FEATURE_HPP_
