#ifndef CUIMG_FFAST382SL_FEATURE_HPP_
# define CUIMG_FFAST382SL_FEATURE_HPP_

# include <cuda.h>
# include <cuda_runtime.h>
# include <host_defines.h>
# include <cudaGL.h>
# include <cuda_gl_interop.h>

# include <cuimg/copy.h>
# include <cuimg/neighb2d_data.h>
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

  #define s1_tex UNIT_STATIC(hawz453xb)
  #define s2_tex UNIT_STATIC(xyh45gtk)

  texture<float, cudaTextureType2D, cudaReadModeElementType> s1_tex;
  texture<float, cudaTextureType2D, cudaReadModeElementType> s2_tex;

  // inline
  // __host__ __device__
  // float distance_mean(const dffast382sl& a, const dffast382sl& b)
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


#if __CUDA_ARCH__ >= 200
  inline
  __host__ __device__
  float distance_mean_linear(const dffast382sl& a, const dffast382sl& b)
  {
    int d = 0;
    for (char i = 0; i < 16; i++)
      d += ::abs(int(a[i]) - int(b[i]));

    return d / (255.f * 16.f);
  }
#else
  inline
  __host__ __device__
  float distance_mean_linear(const dffast382sl& a, const dffast382sl& b)
  {
    float d = 0;
    for (char i = 0; i < 16; i++)
      d += ::abs(float(a[i]) - float(b[i]));

    return d / (255.f * 16.f);
  }
#endif

  inline
  __host__ __device__
  float distance_mean_linear_s2(const dffast382sl& a, const dffast382sl& b)
  {
    int d = 0;
    for (unsigned i = 8; i < 16; i++)
    {
      int tmp = int(a.distances[i]) - int(b.distances[i]);
      d += tmp * tmp;
    }

    return ::sqrt(float(d)) / ::sqrt(255.f * 8.f);
  }

  inline
  __host__ __device__
  float distance_max_linear(const dffast382sl& a, const dffast382sl& b)
  {
    int d = 0;
    for (unsigned i = 0; i < 16; i++)
    {
      int tmp = ::abs(int(a.distances[i]) - int(b.distances[i]));
      if (tmp > d)
        d = tmp;
    }

    return d;
  }


  inline
  __host__ __device__
  float distance_mean_s2(const dffast382sl& a, const dffast382sl& b)
  {
    int d = 0;
    for (unsigned i = 8; i < 16; i++)
    {
      int tmp = int(a.distances[i]) - int(b.distances[i]);
      d += tmp * tmp;
    }

    return ::sqrt(float(d)) / ::sqrt(255.f * 8.f);
  }


  // inline
  // __host__ __device__
  // float distance_min(const dffast382sl& a, const dffast382sl& b)
  // {
  //   if (::abs(a.pertinence - b.pertinence) > 0.1f ||
  //       b.pertinence < 0.15f ||
  //       a.pertinence < 0.15f)
  //     return 99999.f;
  //   else
  //   {
  //     int d = 9999999;
  //     for (unsigned i = 0; i < 16; i++)
  //     {
  //       int tmp = ::abs(int(a.distances[i]) - int(b.distances[i]));
  //       if (tmp < d)
  //         d = tmp;
  //     }

  //     return d;
  //   }
  // }

  // inline
  // __host__ __device__
  // float distance_max(const dffast382sl& a, const dffast382sl& b)
  // {
  //   if (::abs(a.pertinence - b.pertinence) > 0.1f ||
  //       b.pertinence < 0.15f ||
  //       a.pertinence < 0.15f)
  //     return 99999.f;
  //   else
  //   {
  //     int d = 0;
  //     for (unsigned i = 0; i < 16; i++)
  //     {
  //       int tmp = ::abs(int(a.distances[i]) - int(b.distances[i]));
  //       if (tmp > d)
  //         d = tmp;
  //     }

  //     return d;
  //   }
  // }

  __host__ __device__ inline
  dffast382sl operator+(const dffast382sl& a, const dffast382sl& b)
  {
    dffast382sl res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] + b.distances[i];
    //res.pertinence = a.pertinence + b.pertinence;
    return res;
  }

  __host__ __device__ inline
  dffast382sl operator-(const dffast382sl& a, const dffast382sl& b)
  {
    dffast382sl res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] - b.distances[i];
    //res.pertinence = a.pertinence - b.pertinence;
    return res;
  }

  template <typename S>
  __host__ __device__ inline
  dffast382sl operator/(const dffast382sl& a, const S& s)
  {
    dffast382sl res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] / s;
    //res.pertinence = a.pertinence / s;
    return res;
  }

  template <typename S>
  __host__ __device__ inline
  dffast382sl operator*(const dffast382sl& a, const S& s)
  {
    dffast382sl res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] * s;
    //res.pertinence = a.pertinence * s;
    return res;
  }

  __host__ __device__ inline
  dffast382sl weighted_mean(const dffast382sl& a, float aw, const dffast382sl& b, float bw)
  {
    dffast382sl res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = (float(a.distances[i]) * aw + float(b.distances[i]) * bw) / (aw + bw);
    //res.pertinence = (a.pertinence * aw + b.pertinence * bw) / (aw + bw);
    return res;
  }

   __constant__ const int circle_r3_ffast38s1[8][2] = {
     {-3, 0}, {-2, 2},
     { 0, 3}, { 2, 2},
     { 3, 0}, { 2,-2},
     { 0,-3}, {-2,-2}
   };
   __constant__ const int circle_r3_ffast38s2[8][2] = {
     {-9, 0}, {-6, 6},
     { 0, 9}, { 6, 6},
     { 9, 0}, { 6,-6},
     { 0,-9}, {-6,-6}
   };
  /*
  __constant__ const int circle_r3_ffast382sl[8][2] = {
    {-1, 0}, {-1, 1},
    { 0, 1}, { 1, 1},
    { 1, 0}, { 1,-1},
    { 0,-1}, {-1,-1}
  };
  */


  template <typename V>
  __global__ void FFAST382SL(kernel_image2d<i_float4> frame_color,
                           kernel_image2d<V> frame_s1,
                           kernel_image2d<V> frame_s2,
                           kernel_image2d<dffast382sl> out,
                           kernel_image2d<i_float1> pertinence,
                           float grad_thresh)
  {
    point2d<int> p = thread_pos2d();
    if (!frame_s1.has(p))//; || track(p).x == 0)
      return;


    if (p.row() < 6 || p.row() >= pertinence.domain().nrows() - 6 ||
        p.col() < 6 || p.col() >= pertinence.domain().ncols() - 6)
    {
      pertinence(p).x = 0.f;
      return;
    }

    dffast382sl distances;

    float pv;

    {
      float min_diff = 9999999.f;
      float max_single_diff = 0.f;
      pv = tex2D(s1_tex, p);
      int sign = 0;
      for(int i = 0; i < 8; i++)
      {

        float v1 = tex2D(s1_tex,
                         p.col() + circle_r3[i][1],
                         p.row() + circle_r3[i][0]
          );

        float v2 = tex2D(s1_tex,
                         p.col() + circle_r3[(i+8)][1],
                         p.row() + circle_r3[(i+8)][0]);

        if (!(i % 2))
        {
            distances[i/2] = (v1 * 255);
            distances[i/2 + 4] = (v2 * 255);
        }

        {
          float diff = pv -
            (v1 + v2) / 2.f;
          float adiff = ::abs(diff);

          if (adiff < min_diff)
            min_diff = adiff;

          if (max_single_diff < adiff) max_single_diff = adiff;

        }
      }

      pv = tex2D(s2_tex, i_int2(p)/2);
      float min_diff_large = 9999999.f;
      float max_single_diff_large = 0.f;
      //int min_orientation_large;
      for(int i = 0; i < 8; i++)
      {

        float v1 = tex2D(s2_tex,
                         p.col()/2 + circle_r3[i][1],
                         p.row()/2 + circle_r3[i][0]);

        float v2 = tex2D(s2_tex,
                         p.col()/2 + circle_r3[(i+8)][1],
                         p.row()/2 + circle_r3[(i+8)][0]);

        if (!(i % 2))
        {
            distances[8 + i/2] = (v1 * 255);
            distances[8 + i/2 + 4] = (v2 * 255);
        }

        {
          float diff = pv - (v1 + v2) / 2.f;
          float adiff = ::abs(diff);

          if (adiff < min_diff_large)
          {
            min_diff_large = adiff;
            //if (min_diff_large < 0.01) break;
          }

          if (max_single_diff_large < adiff) max_single_diff_large = adiff;
        }

      }

      if (min_diff < min_diff_large)
      {
        min_diff = min_diff_large;
        max_single_diff = max_single_diff_large;
      }

      if (max_single_diff >= grad_thresh)
      {
        min_diff = min_diff / max_single_diff;
      }
      else
        min_diff = 0;

      pertinence(p) = min_diff;
      out(p) = distances;

    }

  }

  template <typename T>
  __global__  void dffast382sl_to_color(kernel_image2d<dffast382sl> in,
                                  kernel_image2d<i_float4> out)
  {
    point2d<int> p = thread_pos2d();
    if (!in.has(p))
      return;

    i_float4 res;
    for (unsigned i = 0; i < 8; i+=2)
      //res[i/2] = (::abs(in(p).distances[i]) + ::abs(in(p).distances[i+1])) / (2*127.f);
      res[i/2] = (::abs(float(in(p).distances[i]) / 255.f) + ::abs(float(in(p).distances[i+1]) / 255.f)) / 2;
    res.w = 1.f;
    out(p) = res;
  }

  inline
  ffast382sl_feature::ffast382sl_feature(const domain_t& d)
    : gl_frame_(d),
      blurred_s1_(d),
      blurred_s2_(d),
      tmp_(d),
      pertinence_(d),
      pertinence2_(d),
      f1_(d),
      f2_(d),
      ffast382sl_color_(d),
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
  ffast382sl_feature::update(const image2d_f1& in, const image2d_f1& in_s2)
  {
    swap_buffers();
    dim3 dimblock(32, 16, 1);
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    // local_jet_static_<0, 0, 1, 3>::run(in, blurred_s1_, tmp_);
    // local_jet_static_<0, 0, 2, 6>::run(in, blurred_s2_, tmp_);

    // local_jet_static2_<0,0,1, 0,0,2, 6>::run(in, blurred_s1_, blurred_s2_, tmp_, pertinence2_);

    // bindTexture2d(blurred_s1_, s1_tex);
    // bindTexture2d(blurred_s2_, s2_tex);

    bindTexture2d(in, s1_tex);
    bindTexture2d(in_s2, s2_tex);

    grad_thresh = Slider("grad_thresh").value() / 100.f;
    FFAST382SL<i_float1><<<dimgrid, dimblock>>>
      (color_blurred_, in, in_s2, *f_, pertinence_, grad_thresh);

    // filter_pertinence<i_float1><<<dimgrid, dimblock>>>
    //   (pertinence_, pertinence2_);
    // copy(pertinence2_, pertinence_);

    cudaUnbindTexture(s1_tex);
    cudaUnbindTexture(s2_tex);
    check_cuda_error();
  }

  inline
  const image2d_f4&
  ffast382sl_feature::feature_color() const
  {
    return ffast382sl_color_;
  }

  inline
  void
  ffast382sl_feature::swap_buffers()
  {
    std::swap(f_prev_, f_);
  }

  inline
  const ffast382sl_feature::domain_t&
  ffast382sl_feature::domain() const
  {
    return f1_.domain();
  }

  inline
  image2d_D&
  ffast382sl_feature::previous_frame()
  {
    return *f_prev_;
  }

  inline
  image2d_D&
  ffast382sl_feature::current_frame()
  {
    return *f_;
  }

  inline
  image2d_f1&
  ffast382sl_feature::pertinence()
  {
    return pertinence_;
  }

  inline
  kernel_ffast382sl_feature::kernel_ffast382sl_feature(ffast382sl_feature& f)
    : pertinence_(f.pertinence()),
      f_prev_(f.previous_frame()),
      f_(f.current_frame())
  {
  }


  // inline
  // __device__ float
  // kernel_ffast382sl_feature::distance(const point2d<int>& p_prev,
  //                               const point2d<int>& p_cur)
  // {
  //   return cuimg::distance_mean(f_prev_(p_prev), f_(p_cur));
  // }

  // inline
  // __device__ float
  // kernel_ffast382sl_feature::distance(const dffast382sl& a,
  //                               const dffast382sl& b)
  // {
  //   return cuimg::distance_mean(a, b);
  // }


  inline
  __device__ float
  kernel_ffast382sl_feature::distance_linear_s2(const dffast382sl& a,
                                const dffast382sl& b)
  {
    return cuimg::distance_mean_linear_s2(a, b);
  }


  inline
  __device__ float
  kernel_ffast382sl_feature::distance_linear(const point2d<int>& p_prev,
                                            const point2d<int>& p_cur)
  {
    return cuimg::distance_mean_linear(f_(p_prev), f_(p_cur));
  }

  inline
  __device__ float
  kernel_ffast382sl_feature::distance_linear(const dffast382sl& a,
                                const dffast382sl& b)
  {
    return cuimg::distance_mean_linear(a, b);
  }


  inline
  __device__ dffast382sl
  kernel_ffast382sl_feature::weighted_mean(const dffast382sl& a, float aw,
                                          const point2d<int>& n, float bw)
  {
    return new_state(n);
  }

  inline
  __device__ float kernel_ffast382sl_feature::distance_linear(const dffast382sl& a,
                                                             const point2d<int>& n)
  {
    return cuimg::distance_mean_linear(a, f_(n));
  }

  inline
  __device__ float
  kernel_ffast382sl_feature::distance_s2(const dffast382sl& a,
                                const dffast382sl& b)
  {
    return cuimg::distance_mean_s2(a, b);
  }

  // inline __device__
  // kernel_image2d<dffast382sl>&
  // kernel_ffast382sl_feature::previous_frame()
  // {
  //   return f_prev_;
  // }

  inline __device__
  kernel_image2d<dffast382sl>&
  kernel_ffast382sl_feature::current_frame()
  {
    return f_;
  }

  inline __device__
  kernel_image2d<i_float1>&
  kernel_ffast382sl_feature::pertinence()
  {
    return pertinence_;
  }

  inline
  __device__ dffast382sl
  kernel_ffast382sl_feature::new_state(const point2d<int>& n)
  {
    return f_(n);
  }

  inline
  void
  ffast382sl_feature::display() const
  {
#ifdef WITH_DISPLAY
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(pertinence_.domain(), dimblock);

    dffast382sl_to_color<int><<<dimgrid, dimblock>>>(*f_, ffast382sl_color_);
    ImageView("test") <<= dg::dl() - gl_frame_ - blurred_s1_ - blurred_s2_ - pertinence_ - ffast382sl_color_;
#endif
  }
}

#endif // ! CUIMG_FFAST382SL_FEATURE_HPP_
