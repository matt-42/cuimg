#ifndef CUIMG_BFAST_FEATURE_HPP_
# define CUIMG_BFAST_FEATURE_HPP_

# include <cuda_runtime.h>
# include <cuimg/gpu/local_jet_static.h>
# include <cuimg/dsl/get_comp.h>
# include <cuimg/dsl/binary_div.h>
# include <cuimg/dsl/binary_add.h>
# include <cuimg/tracking/fast_tools.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_1.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_2.h>

# include <cuimg/dige.h>

# include <dige/widgets/image_view.h>

using dg::dl;
using namespace dg::widgets;

namespace cuimg
{

  inline
  __host__ __device__
  float distance(const dbfast& a, const dbfast& b)
  {
     if (a.intensity <= 0.05f || b.intensity <= 0.05f ||
        b.sign != a.sign)
      return 9999999.f;

    // float d = ::abs((a.orientation - b.orientation) % 4) / 4.f;
    // float e = ::abs(a.intensity - b.intensity) * 2.f;
    // float f = ::abs(a.max_diff - b.max_diff) * 2.f;

    int s = 0;
    unsigned short at = a.tests;
    unsigned short bt = b.tests;
    for (unsigned i = 0; i < 16; i++)
    {
      if ((at & 1) != (bt & 1)) s++;
      at >>= 1;
      bt >>= 1;
    }
    return (norml2(a.color - b.color) + s / 16.f) / 2.f;
    //return (s / 16.f);
  }

  template <typename V>
  __global__ void BFAST(kernel_image2d<i_float4> color_frame,
                       kernel_image2d<V> frame,
                       kernel_image2d<dbfast> out,
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
      unsigned short tests = 0;
      bool sign = false;

      for(unsigned i = 0; i < 16; i++)
      {
        point2d<int> n1(p.row() + circle_r3[i][0],
          p.col() + circle_r3[i][1]);
        if (frame.has(n1))
          tests += frame(p).x < frame(n1).x && ::abs(frame(p).x - frame(n1).x);
        tests <<= 1;
      }

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
        dbfast res;
        res.max_diff = 0;
        res.intensity = 0;
        res.orientation = 0;
        res.tests = 0;
        out(p) = res;
        pertinence(p) = 0.f;
      }
      else
      {
        dbfast res;
        res.max_diff = max_single_diff;
        res.intensity = min_diff / max_single_diff;
        res.orientation = min_orientation;
        res.sign = sign;
        //unsigned short tf = tests >> min_orientation;
        //unsigned short tl = tests << (16 - min_orientation);
        //res.tests = tl + tf;
        res.tests = tests;
        res.color = color_frame(p);
        out(p) = res;
        pertinence(p) = min(min_diff / max_single_diff, max_single_diff);
      }
    }

  }

  template <typename T>
  __global__  void dbfast_to_color(kernel_image2d<dbfast> in,
                                  kernel_image2d<i_float4> out)
  {
    point2d<int> p = thread_pos2d();
    if (!in.has(p))
      return;

    out(p) = int_to_color(in(p).orientation*16) * in(p).intensity;
  }


  inline
  bfast_feature::bfast_feature(const domain_t& d)
    : gl_frame_(d),
      blurred_(d),
      tmp_(d),
      pertinence_(d),
      f1_(d),
      f2_(d),
      bfast_color_(d),
      grad_thresh(0.3f)
  {
    f_prev_ = &f1_;
    f_ = &f2_;
  }

  inline
  void
  bfast_feature::update(const image2d_f4& in)
  {
    gl_frame_ = (get_x(in) + get_y(in) + get_z(in)) / 3.f;
    swap_buffers();
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    local_jet_static_<0, 0, 2, 6>::run(gl_frame_, blurred_, tmp_);


    grad_thresh = Slider("grad_thresh").value() / 100.f;
    BFAST<i_float1><<<dimgrid, dimblock>>>
      (in, blurred_, *f_, pertinence_, grad_thresh);

    dbfast_to_color<int><<<dimgrid, dimblock>>>(*f_, bfast_color_);
    ImageView("test") <<= dg::dl() - gl_frame_ - pertinence_ - bfast_color_;

    check_cuda_error();
  }

  inline
  void
  bfast_feature::swap_buffers()
  {
    std::swap(f_prev_, f_);
  }

  inline
  const bfast_feature::domain_t&
  bfast_feature::domain() const
  {
    return f1_.domain();
  }

  inline
  image2d_D&
  bfast_feature::previous_frame()
  {
    return *f_prev_;
  }

  inline
  image2d_D&
  bfast_feature::current_frame()
  {
    return *f_;
  }

  inline
  image2d_f1&
  bfast_feature::pertinence()
  {
    return pertinence_;
  }

  inline
  kernel_bfast_feature::kernel_bfast_feature(bfast_feature& f)
    : pertinence_(f.pertinence()),
      f_prev_(f.previous_frame()),
      f_(f.current_frame())
  {
  }

  inline
  __device__ float
  kernel_bfast_feature::distance(const point2d<int>& p_prev,
                                const point2d<int>& p_cur)
  {
    return cuimg::distance(f_prev_(p_prev), f_(p_cur));
  }

  inline
  __device__ float
  kernel_bfast_feature::distance(const dbfast& a,
                                const dbfast& b)
  {
    return cuimg::distance(a, b);
  }

  inline __device__
  kernel_image2d<dbfast>&
  kernel_bfast_feature::previous_frame()
  {
    return f_prev_;
  }

  inline __device__
  kernel_image2d<dbfast>&
  kernel_bfast_feature::current_frame()
  {
    return f_;
  }

  inline __device__
  kernel_image2d<i_float1>&
  kernel_bfast_feature::pertinence()
  {
    return pertinence_;
  }

}

#endif // ! CUIMG_BFAST_FEATURE_HPP_
