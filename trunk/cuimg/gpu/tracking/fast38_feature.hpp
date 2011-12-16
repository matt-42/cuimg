#ifndef CUIMG_FAST38_FEATURE_HPP_
# define CUIMG_FAST38_FEATURE_HPP_

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
  float distance_mean(const dfast38& a, const dfast38& b)
  {
    if (::abs(a.pertinence - b.pertinence) > 0.1f ||
        b.pertinence < 0.15f ||
        a.pertinence < 0.15f)
      return 99999.f;
    else
    {
      float d = 0.f;
      for (unsigned i = 0; i < 8; i++)
      {
        float tmp = a.distances[i] - b.distances[i];
        d += tmp * tmp;
      }

      return ::sqrt(d) / ::sqrt(8.f);
    }
  }


  inline
  __host__ __device__
  float distance_min(const dfast38& a, const dfast38& b)
  {
    if (::abs(a.pertinence - b.pertinence) > 0.1f ||
        b.pertinence < 0.15f ||
        a.pertinence < 0.15f)
      return 99999.f;
    else
    {
      float d = 9999999.f;
      for (unsigned i = 0; i < 8; i++)
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
  float distance_max(const dfast38& a, const dfast38& b)
  {
    if (::abs(a.pertinence - b.pertinence) > 0.1f ||
        b.pertinence < 0.15f ||
        a.pertinence < 0.15f)
      return 99999.f;
    else
    {
      float d = 0.f;
      for (unsigned i = 0; i < 8; i++)
      {
        float tmp = ::abs(a.distances[i] - b.distances[i]);
        if (tmp > d)
          d = tmp;
      }

      return d;
    }
  }

  __host__ __device__ inline
  dfast38 operator+(const dfast38& a, const dfast38& b)
  {
    dfast38 res;
    for (unsigned i = 0; i < 8; i++)
      res.distances[i] = a.distances[i] + b.distances[i];
    res.pertinence = a.pertinence + b.pertinence;
    return res;
  }

  __host__ __device__ inline
  dfast38 operator-(const dfast38& a, const dfast38& b)
  {
    dfast38 res;
    for (unsigned i = 0; i < 8; i++)
      res.distances[i] = a.distances[i] - b.distances[i];
    res.pertinence = a.pertinence - b.pertinence;
    return res;
  }

  template <typename S>
  __host__ __device__ inline
  dfast38 operator/(const dfast38& a, const S& s)
  {
    dfast38 res;
    for (unsigned i = 0; i < 8; i++)
      res.distances[i] = a.distances[i] / s;
    res.pertinence = a.pertinence / s;
    return res;
  }

  template <typename S>
  __host__ __device__ inline
  dfast38 operator*(const dfast38& a, const S& s)
  {
    dfast38 res;
    for (unsigned i = 0; i < 8; i++)
      res.distances[i] = a.distances[i] * s;
    res.pertinence = a.pertinence * s;
    return res;
  }


   __constant__ const int circle_r3_fast38[8][2] = {
     {-3, 0}, {-2, 2},
     { 0, 3}, { 2, 2},
     { 3, 0}, { 2,-2},
     { 0,-3}, {-2,-2}
   };
  /*
  __constant__ const int circle_r3_fast38[8][2] = {
    {-1, 0}, {-1, 1},
    { 0, 1}, { 1, 1},
    { 1, 0}, { 1,-1},
    { 0,-1}, {-1,-1}
  };
  */
  template <typename V>
  __global__ void FAST38(kernel_image2d<i_float4> frame_color,
                       kernel_image2d<V> frame,
                       kernel_image2d<dfast38> out,
                       kernel_image2d<i_float1> pertinence,
                       float grad_thresh)
  {
    point2d<int> p = thread_pos2d();
    if (!frame.has(p))//; || track(p).x == 0)
      return;

    float distances[8];
    float plj = frame(p).x;
    {
      for(unsigned i = 0; i < 8; i++)
      {
        point2d<int> n1(p.row() + circle_r3_fast38[i][0],
                        p.col() + circle_r3_fast38[i][1]);
        if (frame.has(n1))
          //distances[i] = norml2(frame(n1) - frame(p));
          distances[i] = frame(n1) - frame(p);
          //distances[i] = frame(n1);
          //distances[i] = norml2(frame_color(n1) - frame_color(p)) / 2.f;
        else
          distances[i] = 0.f;
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

        if (frame.has(n1) && frame.has(n2))
        {
          float diff = plj - (frame(n1).x + frame(n2).x) / 2.f;
          float adiff = ::abs(diff);
          if (adiff < min_diff)
          {
            min_diff = adiff;
            min_orientation = i;
            sign = diff > 0;
          }
          float sd = ::max(::abs(plj - frame(n1).x), ::abs(plj - frame(n2).x));
          if (max_single_diff < sd) max_single_diff = sd;

        }
      }

      for(unsigned i = 0; i < 8; i++)
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
  __global__  void dfast38_to_color(kernel_image2d<dfast38> in,
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
  fast38_feature::fast38_feature(const domain_t& d)
    : gl_frame_(d),
      blurred_(d),
      tmp_(d),
      pertinence_(d),
      f1_(d),
      f2_(d),
      fast38_color_(d),
      color_blurred_(d),
      color_tmp_(d),
      grad_thresh(0.3f)
  {
    f_prev_ = &f1_;
    f_ = &f2_;
  }

  inline
  void
  fast38_feature::update(const image2d<i_float4>& in)
  {
    gl_frame_ = (get_x(in) + get_y(in) + get_z(in)) / 3.f;
    update(gl_frame_);
  }

  inline
  void
  fast38_feature::update(const image2d<i_float1>& gl_frame)
  {
    swap_buffers();
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(gl_frame.domain(), dimblock);

    local_jet_static_<0, 0, 2, 10>::run(gl_frame, blurred_, tmp_);
    //local_jet_static_<0, 0, 1, 5>::run(in, color_blurred_, color_tmp_);


    grad_thresh = Slider("grad_thresh").value() / 100.f;
    FAST38<i_float1><<<dimgrid, dimblock>>>
      (color_blurred_, blurred_, *f_, pertinence_, grad_thresh);

#ifdef WITH_DISPLAY
    dfast38_to_color<int><<<dimgrid, dimblock>>>(*f_, fast38_color_);
    ImageView("test") <<= dg::dl() - gl_frame - pertinence_ - fast38_color_;
#endif

    check_cuda_error();
  }

  inline
  const image2d<i_float4>&
  fast38_feature::feature_color() const
  {
    return fast38_color_;
  }

  inline
  void
  fast38_feature::swap_buffers()
  {
    std::swap(f_prev_, f_);
  }

  inline
  const fast38_feature::domain_t&
  fast38_feature::domain() const
  {
    return f1_.domain();
  }

  inline
  image2d<dfast38>&
  fast38_feature::previous_frame()
  {
    return *f_prev_;
  }

  inline
  image2d<dfast38>&
  fast38_feature::current_frame()
  {
    return *f_;
  }

  inline
  image2d<i_float1>&
  fast38_feature::pertinence()
  {
    return pertinence_;
  }

  inline
  kernel_fast38_feature::kernel_fast38_feature(fast38_feature& f)
    : pertinence_(f.pertinence()),
      f_prev_(f.previous_frame()),
      f_(f.current_frame())
  {
  }

  inline
  __device__ float
  kernel_fast38_feature::distance(const point2d<int>& p_prev,
                                const point2d<int>& p_cur)
  {
    return cuimg::distance_max(f_prev_(p_prev), f_(p_cur));
  }

  inline
  __device__ float
  kernel_fast38_feature::distance(const dfast38& a,
                                const dfast38& b)
  {
    return cuimg::distance_max(a, b);
  }

  inline __device__
  kernel_image2d<dfast38>&
  kernel_fast38_feature::previous_frame()
  {
    return f_prev_;
  }

  inline __device__
  kernel_image2d<dfast38>&
  kernel_fast38_feature::current_frame()
  {
    return f_;
  }

  inline __device__
  kernel_image2d<i_float1>&
  kernel_fast38_feature::pertinence()
  {
    return pertinence_;
  }

}

#endif // ! CUIMG_FAST38_FEATURE_HPP_
