#ifndef CUIMG_FAST3_FEATURE_HPP_
# define CUIMG_FAST3_FEATURE_HPP_

# include <cuda_runtime.h>
# include <cuimg/gpu/local_jet_static.h>
# include <cuimg/dsl/binary_div.h>
# include <cuimg/dsl/binary_add.h>
# include <cuimg/dsl/get_comp.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_1.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_2.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_4.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_3.h>
# include <cuimg/tracking/fast_tools.h>

# include <cuimg/dige.h>

#include <dige/widgets/image_view.h>

using dg::dl;
using namespace dg::widgets;

namespace cuimg
{

  inline
  __host__ __device__
  float distance(const dfast3& a, const dfast3& b)
  {
    if (::abs(a.pertinence - b.pertinence) > 0.1f)
      return 99999.f;
    else
      return norml2(a.distances - b.distances) / ::sqrt(4.f);
  }

  __host__ __device__ inline
  dfast3 operator+(const dfast3& a, const dfast3& b)
  {
    dfast3 res;
    res.distances = a.distances + b.distances;
    res.pertinence = a.pertinence + b.pertinence;
    return res;
  }

  __host__ __device__ inline
  dfast3 operator-(const dfast3& a, const dfast3& b)
  {
    dfast3 res;
    res.distances = a.distances - b.distances;
    res.pertinence = a.pertinence - b.pertinence;
    return res;
  }

  template <typename S>
  __host__ __device__ inline
  dfast3 operator/(const dfast3& a, const S& s)
  {
    dfast3 res;
    res.distances = a.distances / s;
    res.pertinence = a.pertinence / s;
    return res;
  }

  template <typename S>
  __host__ __device__ inline
  dfast3 operator*(const dfast3& a, const S& s)
  {
    dfast3 res;
    res.distances = a.distances * s;
    res.pertinence = a.pertinence * s;
    return res;
  }


  __constant__ const int circle_r3_fast3[4][2] = {
    {-3, 0},
    { 0, 3},
    { 3, 0},
    { 0,-3}
  };

  template <typename V>
  __global__ void FAST3(kernel_image2d<V> frame,
                       kernel_image2d<dfast3> out,
                       kernel_image2d<i_float1> pertinence,
                       float grad_thresh)
  {
    point2d<int> p = thread_pos2d();
    if (!frame.has(p))//; || track(p).x == 0)
      return;

    i_float4 distances;
    float plj = frame(p).x;
    {
      for(unsigned i = 0; i < 4; i++)
      {
        point2d<int> n1(p.row() + circle_r3_fast3[i][0],
                        p.col() + circle_r3_fast3[i][1]);
        if (frame.has(n1))
          //distances[i] = norml2(frame(n1) - frame(p));
          //distances[i] = norml2(frame(n1)) - norml2(frame(p));
          //distances[i] = frame(n1) - frame(p);
          distances[i] = frame(n1);
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

      out(p).distances = distances / max_single_diff;
      if (max_single_diff >= grad_thresh)
      {
        out(p).pertinence = min_diff / max_single_diff;
        pertinence(p) = min_diff / max_single_diff;//min(min_diff / max_single_diff, max_single_diff);
      }
      else
      {
        pertinence(p) = 0.f;
        out(p).pertinence = 0.f;
      }

    }

  }

  template <typename T>
  __global__  void dfast3_to_color(kernel_image2d<dfast3> in,
                                  kernel_image2d<i_float4> out)
  {
    point2d<int> p = thread_pos2d();
    if (!in.has(p))
      return;

    i_float4 res = in(p).distances;
    res.w = 1.f;
    out(p) = res;
  }

  inline
  fast3_feature::fast3_feature(const domain_t& d)
    : gl_frame_(d),
      blurred_(d),
      tmp_(d),
      pertinence_(d),
      f1_(d),
      f2_(d),
      fast3_color_(d),
      grad_thresh(0.3f)
  {
    f_prev_ = &f1_;
    f_ = &f2_;
  }

  inline
  void
  fast3_feature::update(const image2d_f4& in)
  {
    gl_frame_ = (get_x(in) + get_y(in) + get_z(in)) / 3.f;
    swap_buffers();
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    local_jet_static_<0, 0, 1, 5>::run(gl_frame_, blurred_, tmp_);


    grad_thresh = Slider("grad_thresh").value() / 100.f;
    FAST3<i_float1><<<dimgrid, dimblock>>>
      (blurred_, *f_, pertinence_, grad_thresh);

    dfast3_to_color<int><<<dimgrid, dimblock>>>(*f_, fast3_color_);

    ImageView("test") <<= dg::dl() - gl_frame_ - pertinence_ - fast3_color_;

    check_cuda_error();
  }

  inline
  const image2d_f4&
  fast3_feature::feature_color() const
  {
    return fast3_color_;
  }

  inline
  void
  fast3_feature::swap_buffers()
  {
    std::swap(f_prev_, f_);
  }

  inline
  const fast3_feature::domain_t&
  fast3_feature::domain() const
  {
    return f1_.domain();
  }

  inline
  image2d_D&
  fast3_feature::previous_frame()
  {
    return *f_prev_;
  }

  inline
  image2d_D&
  fast3_feature::current_frame()
  {
    return *f_;
  }

  inline
  image2d_f1&
  fast3_feature::pertinence()
  {
    return pertinence_;
  }

  inline
  kernel_fast3_feature::kernel_fast3_feature(fast3_feature& f)
    : pertinence_(f.pertinence()),
      f_prev_(f.previous_frame()),
      f_(f.current_frame())
  {
  }

  inline
  __device__ float
  kernel_fast3_feature::distance(const point2d<int>& p_prev,
                                const point2d<int>& p_cur)
  {
    return cuimg::distance(f_prev_(p_prev), f_(p_cur));
  }

  inline
  __device__ float
  kernel_fast3_feature::distance(const dfast3& a,
                                const dfast3& b)
  {
    return cuimg::distance(a, b);
  }

  inline __device__
  kernel_image2d<dfast3>&
  kernel_fast3_feature::previous_frame()
  {
    return f_prev_;
  }

  inline __device__
  kernel_image2d<dfast3>&
  kernel_fast3_feature::current_frame()
  {
    return f_;
  }

  inline __device__
  kernel_image2d<i_float1>&
  kernel_fast3_feature::pertinence()
  {
    return pertinence_;
  }

}

#endif // ! CUIMG_FAST3_FEATURE_HPP_
