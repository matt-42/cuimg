#ifndef CUIMG_FAST_FEATURE_HPP_
# define CUIMG_FAST_FEATURE_HPP_

# include <cuda_runtime.h>
# include <cuimg/gpu/local_jet_static.h>
# include <cuimg/dsl/binary_div.h>
# include <cuimg/dsl/binary_add.h>
# include <cuimg/dsl/get_comp.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_1.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_2.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_4.h>
# include <cuimg/gpu/tracking/fast_tools.h>

# include <cuimg/dige.h>

#include <dige/widgets/image_view.h>

using dg::dl;
using namespace dg::widgets;

namespace cuimg
{

  inline
  __host__ __device__
  float distance(const dfast& a, const dfast& b)
  {
    if (a.intensity <= 0.03f || b.intensity <= 0.003f ||
        b.sign != a.sign)
      return 9999999.f;

      float d = (::abs(a.orientation - b.orientation) % 4) / 4.f;
  float e = ::abs(a.intensity - b.intensity) * 5.f;
  float f = ::abs(a.max_diff - b.max_diff) * 5.f;

  //return max(d, max(e, f));
  return max(f, e);
  /*
    return (//::abs((a.orientation - b.orientation) % 4) / 4.f +
            ::abs(a.intensity - b.intensity) * 2.f +
            ::abs(a.max_diff - b.max_diff) * 2.f);
  */}



  template <typename V>
  __global__ void FAST(kernel_image2d<V> frame,
                       kernel_image2d<dfast> out,
                       kernel_image2d<i_float1> pertinence,
                       float grad_thresh)
  {
    point2d<int> p = thread_pos2d();
    if (!frame.has(p))//; || track(p).x == 0)
      return;

    float plj = frame(p).x;
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

      if (max_single_diff < grad_thresh)
      {
        dfast res;
        res.max_diff = 0;
        res.intensity = 0;
        res.orientation = 0;
        out(p) = res;
        pertinence(p) = 0.f;
      }
      else
      {
        dfast res;
        res.max_diff = max_single_diff;
        res.intensity = min_diff / max_single_diff;
        res.orientation = min_orientation;
        res.sign = sign;
        out(p) = res;
        pertinence(p) = min(min_diff / max_single_diff, max_single_diff);
      }
    }

  }

  template <typename T>
  __global__  void dfast_to_color(kernel_image2d<dfast> in,
                                  kernel_image2d<i_float4> out)
  {
    point2d<int> p = thread_pos2d();
    if (!in.has(p))
      return;

    out(p) = int_to_color(in(p).orientation*16) * in(p).intensity;
  }

  template <typename F>
  __global__ void local_maximas(kernel_image2d<F> in,
    kernel_image2d<i_float1> out)
  {
    point2d<int> p = thread_pos2d();
    if (!in.has(p))//; || track(p).x == 0)
      return;

    bool maxima = true;
    float pi = in(p).intensity;
    for_all_in_static_neighb2d(p, n, c8) if (in.has(n))
      if (in(n).intensity >= pi) maxima = false;

    if (maxima) out(p) = 1.f;
    else
      out(p) = 0.f;
  }


  inline
  fast_feature::fast_feature(const domain_t& d)
    : gl_frame_(d),
      blurred_(d),
      tmp_(d),
      pertinence_(d),
      f1_(d),
      f2_(d),
      fast_color_(d),
      grad_thresh(0.3f)
  {
    f_prev_ = &f1_;
    f_ = &f2_;
  }

  inline
  void
  fast_feature::update(const image2d<i_float4>& in)
  {
    gl_frame_ = (get_x(in) + get_y(in) + get_z(in)) / 3.f;
    swap_buffers();
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    local_jet_static_<0, 0, 2, 6>::run(gl_frame_, blurred_, tmp_);


    grad_thresh = Slider("grad_thresh").value() / 100.f;
    FAST<i_float1><<<dimgrid, dimblock>>>
      (blurred_, *f_, pertinence_, grad_thresh);

    local_maximas<dfast><<<dimgrid, dimblock>>>(*f_, pertinence_);
    //dfast_to_color<int><<<dimgrid, dimblock>>>(*f_, fast_color_);
    // ImageView("test") <<= dg::dl() - gl_frame_ - pertinence_ - fast_color_;

    check_cuda_error();
  }

  inline
  void
  fast_feature::swap_buffers()
  {
    std::swap(f_prev_, f_);
  }

  inline
  const fast_feature::domain_t&
  fast_feature::domain() const
  {
    return f1_.domain();
  }

  inline
  image2d<dfast>&
  fast_feature::previous_frame()
  {
    return *f_prev_;
  }

  inline
  image2d<dfast>&
  fast_feature::current_frame()
  {
    return *f_;
  }

  inline
  image2d<i_float1>&
  fast_feature::pertinence()
  {
    return pertinence_;
  }

  inline
  kernel_fast_feature::kernel_fast_feature(fast_feature& f)
    : pertinence_(f.pertinence()),
      f_prev_(f.previous_frame()),
      f_(f.current_frame())
  {
  }

  inline
  __device__ float
  kernel_fast_feature::distance(const point2d<int>& p_prev,
                                const point2d<int>& p_cur)
  {
    return cuimg::distance(f_prev_(p_prev), f_(p_cur));
  }

  inline
  __device__ float
  kernel_fast_feature::distance(const dfast& a,
                                const dfast& b)
  {
    return cuimg::distance(a, b);
  }

  inline __device__
  kernel_image2d<dfast>&
  kernel_fast_feature::previous_frame()
  {
    return f_prev_;
  }

  inline __device__
  kernel_image2d<dfast>&
  kernel_fast_feature::current_frame()
  {
    return f_;
  }

  inline __device__
  kernel_image2d<i_float1>&
  kernel_fast_feature::pertinence()
  {
    return pertinence_;
  }

}

#endif // ! CUIMG_FAST_FEATURE_HPP_
