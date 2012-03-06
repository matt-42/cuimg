#ifndef CUIMG_OFAST382SL_FEATURE_HPP_
# define CUIMG_OFAST382SL_FEATURE_HPP_

# include <cuda.h>
# include <cuda_runtime.h>
# include <host_defines.h>
# include <cudaGL.h>
# include <cuda_gl_interop.h>

# include <cuimg/copy.h>
# include <cuimg/gpu/local_jet_static.h>
# include <cuimg/dsl/binary_div.h>
# include <cuimg/dsl/binary_mul.h>
# include <cuimg/dsl/binary_add.h>
# include <cuimg/dsl/get_comp.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_1.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_2.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_4.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_3.h>
# include <cuimg/meta_gaussian/meta_gaussian_coefs_6.h>
# include <cuimg/tracking/fast_tools.h>
# include <cuimg/gpu/texture.h>

# include <cuimg/dige.h>

#include <dige/widgets/image_view.h>

using dg::dl;
using namespace dg::widgets;

namespace cuimg
{

  #define s1_ofast_tex UNIT_STATIC(hawzxbsdsdf)
  #define s2_ofast_tex UNIT_STATIC(xyhgtkfsdfr)

  texture<float, cudaTextureType2D, cudaReadModeElementType> s1_ofast_tex;
  texture<float, cudaTextureType2D, cudaReadModeElementType> s2_ofast_tex;

  // inline
  // __host__ __device__
  // float distance_mean(const dofast382sl& a, const dofast382sl& b)
  // {
  //   if (::abs(a.pertinence - b.pertinence) > 0.2f ||
  //       b.pertinence < 0.1f ||
  //       a.pertinence < 0.1f
  //       )
  //     return 99999.f;
  //   else
  //   {
  //     int d = 0;
  //     for (unsigned i = 0; i < 16; i++)
  //     {
  //       int tmp = int(a.distances[i]) - int(b.distances[i]);
  //       d += tmp * tmp;
  //     }

  //     return ::sqrt(float(d)) / ::sqrt(255.f * 16.f);
  //   }
  // }


// #if __CUDA_ARCH__ >= 200
//   inline
//   __host__ __device__
//   float distance_mean_linear(const dofast382sl& a, const dofast382sl& b)
//   {
//     int d = 0;
//     for (char i = 0; i < 16; i++)
//       d += ::abs(int(a[i]) - int(b[i]));

//     return d / (255.f * 16.f);
//   }
// #else
//   inline
//   __host__ __device__
//   float distance_mean_linear(const dofast382sl& a, const dofast382sl& b)
//   {
//     float d = 0;
//     for (char i = 0; i < 16; i++)
//       d += ::abs(float(a[i]) - float(b[i]));

//     return d / (255.f * 16.f);
//   }
// #endif

   __constant__ const int circle_r3_ofast38s1[8][2] = {
     {-3, 0}, {-2, 2},
     { 0, 3}, { 2, 2},
     { 3, 0}, { 2,-2},
     { 0,-3}, {-2,-2}
   };
   __constant__ const int circle_r3_ofast38s2[8][2] = {
     {-9, 0}, {-6, 6},
     { 0, 9}, { 6, 6},
     { 9, 0}, { 6,-6},
     { 0,-9}, {-6,-6}
   };
  /*
  __constant__ const int circle_r3_ofast382sl[8][2] = {
    {-1, 0}, {-1, 1},
    { 0, 1}, { 1, 1},
    { 1, 0}, { 1,-1},
    { 0,-1}, {-1,-1}
  };
  */


  template <typename V>
  __global__ void OFAST382SL(kernel_image2d<i_float4> frame_color,
                           kernel_image2d<V> frame_s1,
                           kernel_image2d<V> frame_s2,
                           kernel_image2d<dofast382sl> out,
                           kernel_image2d<i_float1> pertinence,
                           float grad_thresh)
  {
    point2d<int> p = thread_pos2d();
    if (!frame_s1.has(p))//; || track(p).x == 0)
      return;


    if (p.row() < 6 || p.row() >= pertinence.domain().nrows() - 6 ||
        p.col() < 6 || p.col() >= pertinence.domain().ncols() - 6)
    {
      out(p).o1 = 0;
      out(p).o2 = 0;
      pertinence(p).x = 0.f;
      return;
    }

    dofast382sl desc;
    desc.o1 = 0;
    desc.o2 = 0;
    float pv;
    {
      float min_diff = 9999999.f;
      float max_single_diff = 0.f;
      //int min_orientation = 0;
      pv = tex2D(s1_ofast_tex, p);

      for(int i = 0; i < 8; i++)
      {

        float v1 = tex2D(s1_ofast_tex,
                         p.col() + circle_r3[i][1],
                         p.row() + circle_r3[i][0]
          );

        float v2 = tex2D(s1_ofast_tex,
                         p.col() + circle_r3[(i+8)][1],
                         p.row() + circle_r3[(i+8)][0]);

        {
           float diff = pv - (v1 + v2) / 2.f;
          //float diff = ::abs(pv - v1) + ::abs(pv - v2);

          float adiff = ::abs(diff);
          if (adiff < min_diff)
          {
            min_diff = adiff;
          }

          if (max_single_diff < adiff)
          {
            desc.o1 = i;
            max_single_diff = adiff;
          }

        }
      }


      //pv = frame_s2(p);
      pv = tex2D(s2_ofast_tex, p);
      float min_diff_large = 9999999.f;
      float max_single_diff_large = 0.f;
      //int min_orientation_large;
      for(int i = 0; i < 8; i++)
      {

        float v1 = tex2D(s2_ofast_tex,
                         p.col() + 2 * circle_r3[i][1],
                         p.row() + 2 * circle_r3[i][0]);

        float v2 = tex2D(s2_ofast_tex,
                         p.col() + 2 * circle_r3[(i+8)][1],
                         p.row() + 2 * circle_r3[(i+8)][0]);

        {
          //float diff = ::abs(pv - v1) + ::abs(pv - v2);
          float diff = pv - (v1 + v2) / 2.f;
          float adiff = ::abs(diff);
          if (adiff < min_diff_large)
          {
            min_diff_large = adiff;
            //if (min_diff_large < 0.01) break;
          }

          if (max_single_diff_large < adiff)
          //if (max_single_diff_large < )
          {
            desc.o2 = i;
            max_single_diff_large = adiff;
          }
        }

      }

      if (min_diff < min_diff_large)
      {
        min_diff = min_diff_large;
        max_single_diff = max_single_diff_large;
      }

      // desc.o2 = (desc.o1 + 4) & 7;

      if (max_single_diff >= grad_thresh)
        min_diff = min_diff / max_single_diff;

      pertinence(p) = min_diff;
      out(p) = desc;

    }

  }

  template <typename T>
  __global__  void dofast382sl_to_color(kernel_image2d<dofast382sl> in,
                                  kernel_image2d<i_float4> out)
  {
    point2d<int> p = thread_pos2d();
    if (!in.has(p))
      return;

    i_float4 res;
    res.x = in(p).o1;
    res.y = in(p).o2;
    res.w = 1.f;
    out(p) = res;
  }

  inline
  ofast382sl_feature::ofast382sl_feature(const domain_t& d)
    : gl_frame_(d),
      blurred_s1_(d),
      blurred_s2_(d),
      tmp_(d),
      pertinence_(d),
      pertinence2_(d),
      f1_(d),
      f2_(d),
      ofast382sl_color_(d),
      color_blurred_(d),
      color_tmp_(d),
      grad_thresh(0.3f)
  {
    f_prev_ = &f1_;
    f_ = &f2_;
    cudaStreamCreate(&cuda_stream_);
  }

  inline
  void
  ofast382sl_feature::update(const image2d_f4& in)
  {
    gl_frame_ = (get_x(in) + get_y(in) + get_z(in)) / 3.f;
    update(gl_frame_);
  }

  inline
  void
  ofast382sl_feature::update(const image2d_f1& in)
  {
    swap_buffers();
    dim3 dimblock(32, 16, 1);
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    // local_jet_static_<0, 0, 1, 3>::run(in, blurred_s1_, tmp_);
    // local_jet_static_<0, 0, 2, 6>::run(in, blurred_s2_, tmp_);

    local_jet_static2_<0,0,1, 0,0,2, 6>::run(in, blurred_s1_, blurred_s2_, tmp_, pertinence2_);

    bindTexture2d(blurred_s1_, s1_ofast_tex);
    bindTexture2d(blurred_s2_, s2_ofast_tex);

    grad_thresh = Slider("grad_thresh").value() / 100.f;
    OFAST382SL<i_float1><<<dimgrid, dimblock>>>
      (color_blurred_, blurred_s1_, blurred_s2_, *f_, pertinence_, grad_thresh);

    // filter_pertinence<i_float1><<<dimgrid, dimblock>>>
    //   (pertinence_, pertinence2_);
    // copy(pertinence2_, pertinence_);

    cudaUnbindTexture(s1_ofast_tex);
    cudaUnbindTexture(s2_ofast_tex);
    check_cuda_error();
  }

  inline
  const image2d_f4&
  ofast382sl_feature::feature_color() const
  {
    return ofast382sl_color_;
  }

  inline
  void
  ofast382sl_feature::swap_buffers()
  {
    std::swap(f_prev_, f_);
  }

  inline
  const ofast382sl_feature::domain_t&
  ofast382sl_feature::domain() const
  {
    return f1_.domain();
  }

  inline
  image2d_D&
  ofast382sl_feature::previous_frame()
  {
    return *f_prev_;
  }

  inline
  image2d_D&
  ofast382sl_feature::current_frame()
  {
    return *f_;
  }

  inline
  image2d_f1&
  ofast382sl_feature::pertinence()
  {
    return pertinence_;
  }

  inline
  image2d_f1&
  ofast382sl_feature::blurred_s1()
  {
    return blurred_s1_;
  }

  inline
  image2d_f1&
  ofast382sl_feature::blurred_s2()
  {
    return blurred_s2_;
  }

  inline
  kernel_ofast382sl_feature::kernel_ofast382sl_feature(ofast382sl_feature& f)
    : pertinence_(f.pertinence()),
      //f_prev_(f.previous_frame()),
      f_(f.current_frame()),
      blurred_s1_(f.blurred_s1()),
      blurred_s2_(f.blurred_s2())
  {
  }


  // inline
  // __device__ float
  // kernel_ofast382sl_feature::distance(const point2d<int>& p_prev,
  //                               const point2d<int>& p_cur)
  // {
  //   return cuimg::distance_mean(f_prev_(p_prev), f_(p_cur));
  // }

  // inline
  // __device__ float
  // kernel_ofast382sl_feature::distance(const dofast382sl& a,
  //                               const dofast382sl& b)
  // {
  //   return cuimg::distance_mean(a, b);
  // }


  inline
  __device__ float
  kernel_ofast382sl_feature::distance_linear(const point2d<int>& a,
                                             const point2d<int>& b)
  {
    float d = 0;

    dofast382sl da = f_(a);
    // d += ::abs(blurred_s1_(i_int2(a) + i_int2(circle_r3[da.o1])).x * 255 -
    //            blurred_s1_(i_int2(b) + i_int2(circle_r3[da.o1])).x * 255);
    // d += ::abs(blurred_s1_(i_int2(a) + i_int2(circle_r3[da.o1+8])).x * 255 -
    //            blurred_s1_(i_int2(b) + i_int2(circle_r3[da.o1+8])).x * 255);

    // d *= 2;
    // d += ::abs(blurred_s2_(i_int2(a) + i_int2(circle_r3[da.o2]) * 2).x * 255 -
    //            blurred_s2_(i_int2(b) + i_int2(circle_r3[da.o2]) * 2).x * 255);
    // d += ::abs(blurred_s2_(i_int2(a) + i_int2(circle_r3[da.o2+8]) * 2).x * 255 -
    //            blurred_s2_(i_int2(b) + i_int2(circle_r3[da.o2+8]) * 2).x * 255);

    // d = ::max(::max(::abs(blurred_s1_(i_int2(a) + i_int2(circle_r3[da.o1])).x * 255 -
    //                       blurred_s1_(i_int2(b) + i_int2(circle_r3[da.o1])).x * 255),
    //                 ::abs(blurred_s1_(i_int2(a) + i_int2(circle_r3[da.o1+8])).x * 255 -
    //                       blurred_s1_(i_int2(b) + i_int2(circle_r3[da.o1+8])).x * 255)),
    //           ::abs(blurred_s1_(i_int2(a)).x * 255 -
    //                 blurred_s1_(i_int2(b)).x * 255)
    //           );

    // d += ::max(::max(::abs(blurred_s2_(i_int2(a) + i_int2(circle_r3[da.o2]) * 2).x * 255 -
    //                        blurred_s2_(i_int2(b) + i_int2(circle_r3[da.o2]) * 2).x * 255),
    //                  ::abs(blurred_s2_(i_int2(a) + i_int2(circle_r3[da.o2+8]) * 2).x * 255 -
    //                        blurred_s2_(i_int2(b) + i_int2(circle_r3[da.o2+8]) * 2).x * 255)),
    //            ::abs(blurred_s2_(i_int2(a)).x * 255 -
    //                  blurred_s2_(i_int2(b)).x * 255));


    d += ::abs(blurred_s1_(i_int2(a)).x * 255 -
               blurred_s1_(i_int2(b)).x * 255);
    d += ::abs(blurred_s1_(i_int2(a) + i_int2(circle_r3[da.o1])).x * 255 -
               blurred_s1_(i_int2(b) + i_int2(circle_r3[da.o1])).x * 255);
    d += ::abs(blurred_s1_(i_int2(a) + i_int2(circle_r3[da.o1+8])).x * 255 -
               blurred_s1_(i_int2(b) + i_int2(circle_r3[da.o1+8])).x * 255);

    d += ::abs(blurred_s2_(i_int2(a)).x * 255 -
               blurred_s2_(i_int2(b)).x * 255);
    d += ::abs(blurred_s2_(i_int2(a) + i_int2(circle_r3[da.o2]) * 2).x * 255 -
               blurred_s2_(i_int2(b) + i_int2(circle_r3[da.o2]) * 2).x * 255);
    d += ::abs(blurred_s2_(i_int2(a) + i_int2(circle_r3[da.o2+8]) * 2).x * 255 -
               blurred_s2_(i_int2(b) + i_int2(circle_r3[da.o2+8]) * 2).x * 255);


    return d / (255.f * 6.f);
  }

  inline
  __device__ float
  kernel_ofast382sl_feature::distance_linear(const dofast382sl_state& a,
                                             const point2d<int>& n)
  {
    float d = 0;


    i_int2 delta = i_int2(circle_r3[a.o2][0], circle_r3[a.o2][1]) * 2;
    d += ::abs(float(a.state[3]) - blurred_s2_(i_int2(n) + delta).x * 255.f);
    delta = i_int2(circle_r3[a.o2 + 8][0], circle_r3[a.o2 + 8][1]) * 2;
    d += ::abs(float(a.state[4]) - blurred_s2_(i_int2(n) + delta).x * 255.f);
    d += ::abs(float(a.state[5]) - blurred_s2_(i_int2(n)).x * 255.f);

    d *= 1;
    d += ::abs(float(a.state[0]) - blurred_s1_(i_int2(n) + i_int2(circle_r3[a.o1])).x * 255.f);
    d += ::abs(float(a.state[1]) - blurred_s1_(i_int2(n) + i_int2(circle_r3[a.o1+8])).x * 255.f);
    d += ::abs(float(a.state[2]) - blurred_s1_(i_int2(n)).x * 255.f);

    return d / (255.f * 6.f);

  }

  inline
  __device__ dofast382sl_state
  kernel_ofast382sl_feature::new_state(const point2d<int>& n)
  {
    dofast382sl_state res;

    dofast382sl a = f_(n);
    res.state[0] = blurred_s1_(i_int2(n) + i_int2(circle_r3[a.o1])).x * 255;
    res.state[1] = blurred_s1_(i_int2(n) + i_int2(circle_r3[a.o1 + 8])).x * 255;
    res.state[2] = blurred_s1_(i_int2(n)).x * 255;

    i_int2 delta = i_int2(circle_r3[a.o2][0], circle_r3[a.o2][1]) * 2;
    res.state[3] = blurred_s2_(i_int2(n) + delta).x * 255;
    delta = i_int2(circle_r3[a.o2 + 8][0], circle_r3[a.o2 + 8][1]) * 2;
    res.state[4] = blurred_s2_(i_int2(n) + delta).x * 255;
    res.state[5] = blurred_s2_(i_int2(n)).x * 255;

    res.o1 = a.o1;
    res.o2 = a.o2;
    return res;
  }

  inline
  __device__ dofast382sl_state
  kernel_ofast382sl_feature::weighted_mean(const dofast382sl_state& a, float aw,
                                           const point2d<int>& n, float bw)
  {
    dofast382sl_state res = new_state(n);

    // for (int i = 0; i < dofast382sl_state::size; i++)
    //   res.state[i] = (res.state[i] * bw + a.state[i] * aw) / (aw + bw);

    return res;
  }

  inline __device__
  kernel_image2d<dofast382sl>&
  kernel_ofast382sl_feature::current_frame()
  {
    return f_;
  }

  inline __device__
  kernel_image2d<i_float1>&
  kernel_ofast382sl_feature::pertinence()
  {
    return pertinence_;
  }

    inline
  void
  ofast382sl_feature::display() const
  {
#ifdef WITH_DISPLAY
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(pertinence_.domain(), dimblock);

    dofast382sl_to_color<int><<<dimgrid, dimblock>>>(*f_, ofast382sl_color_);
    ImageView("test") <<= dg::dl() - gl_frame_ - blurred_s1_ - blurred_s2_ - pertinence_ - ofast382sl_color_;
#endif
  }
}

#endif // ! CUIMG_OFAST382SL_FEATURE_HPP_
