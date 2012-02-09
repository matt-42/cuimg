#ifndef CUIMG_FAST382SL_FEATURE_HPP_
# define CUIMG_FAST382SL_FEATURE_HPP_

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
# include <cuimg/gpu/tracking/fast_tools.h>
# include <cuimg/gpu/texture.h>

# include <cuimg/dige.h>

#include <dige/widgets/image_view.h>

using dg::dl;
using namespace dg::widgets;

namespace cuimg
{

  #define s1_tex UNIT_STATIC(hawzxb)
  #define s2_tex UNIT_STATIC(xyhgtk)

  texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> s1_tex;
  texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> s2_tex;

  // inline
  // __host__ __device__
  // float distance_mean(const dfast382sl& a, const dfast382sl& b)
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
  float distance_mean_linear(const dfast382sl& a, const dfast382sl& b)
  {
    int d = 0;
    for (char i = 0; i < 16; i++)
      d += ::abs(int(a[i]) - int(b[i]));

    return d / (255.f * 16.f);
  }
#else
  inline
  __host__ __device__
  float distance_mean_linear(const dfast382sl& a, const dfast382sl& b)
  {
    float d = 0;
    for (char i = 0; i < 16; i++)
      d += ::abs(float(a[i]) - float(b[i]));

    return d / (255.f * 16.f);
  }
#endif


  inline
  __host__ __device__
  float distance_mean_linear_s2(const dfast382sl& a, const dfast382sl& b)
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
  float distance_max_linear(const dfast382sl& a, const dfast382sl& b)
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
  float distance_mean_s2(const dfast382sl& a, const dfast382sl& b)
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
  // float distance_min(const dfast382sl& a, const dfast382sl& b)
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
  // float distance_max(const dfast382sl& a, const dfast382sl& b)
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
  dfast382sl operator+(const dfast382sl& a, const dfast382sl& b)
  {
    dfast382sl res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] + b.distances[i];
    //res.pertinence = a.pertinence + b.pertinence;
    return res;
  }

  __host__ __device__ inline
  dfast382sl operator-(const dfast382sl& a, const dfast382sl& b)
  {
    dfast382sl res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] - b.distances[i];
    //res.pertinence = a.pertinence - b.pertinence;
    return res;
  }

  template <typename S>
  __host__ __device__ inline
  dfast382sl operator/(const dfast382sl& a, const S& s)
  {
    dfast382sl res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] / s;
    //res.pertinence = a.pertinence / s;
    return res;
  }

  template <typename S>
  __host__ __device__ inline
  dfast382sl operator*(const dfast382sl& a, const S& s)
  {
    dfast382sl res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] * s;
    //res.pertinence = a.pertinence * s;
    return res;
  }

  __host__ __device__ inline
  dfast382sl weighted_mean(const dfast382sl& a, float aw, const dfast382sl& b, float bw)
  {
    dfast382sl res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = (float(a.distances[i]) * aw + float(b.distances[i]) * bw) / (aw + bw);
    //res.pertinence = (a.pertinence * aw + b.pertinence * bw) / (aw + bw);
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
  __constant__ const int circle_r3_fast382sl[8][2] = {
    {-1, 0}, {-1, 1},
    { 0, 1}, { 1, 1},
    { 1, 0}, { 1,-1},
    { 0,-1}, {-1,-1}
  };
  */


  template <typename V>
  __global__ void filter_pertinence(kernel_image2d<i_float1> pertinence,
                                    kernel_image2d<i_float1> out)
  {
    point2d<int> p = thread_pos2d();
    if (!pertinence.has(p))
      return;



    if (p.row() < 3 || p.row() > pertinence.domain().nrows() - 3 ||
        p.col() < 3 || p.col() > pertinence.domain().ncols() - 3)
    {
      out(p).x = 0.f;
      return;
    }

    float max = pertinence(p).x;

    //for_all_in_static_neighb2d(p, n, c25)
    for(unsigned i = 0; i < 25; i++)
    {
      // point2d<int> n(p.row() + c25[i][0],
      //                p.col() + c25[i][1]);
      float vn = pertinence(p.row() + c25[i][0],
                            p.col() + c25[i][1]);
      if (//pertinence.has(n) &&
          max < vn)
        max = vn;
      //max = ::max(max, pertinence(n).x);
   }

    if (max > 0.3f)
      out(p) = pertinence(p).x;
    else
      out(p) = 0.f;
  }


  template <typename V>
  __global__ void FAST382SL(kernel_image2d<V> frame_s1,
                            kernel_image2d<V> frame_s2,
                            kernel_image2d<dfast382sl> out,
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

    dfast382sl distances;

    float pv;
    /*
    {
      for(unsigned i = 0; i < 8; i++)
      {
        point2d<int> n1(p.row() + circle_r3_fast38s1[i][0],
                        p.col() + circle_r3_fast38s1[i][1]);
        if (frame_s1.has(n1))
          //distances[i] = (frame_s1(n1) - frame_s1(p));
          distances[i] = (frame_s1(n1) * 255);
        else
          distances[i] = 0.f;
      }

      for(unsigned i = 0; i < 8; i++)
      {
        // point2d<int> n1(p.row() + circle_r3_fast38s2[i][0],
        //                 p.col() + circle_r3_fast38s2[i][1]);
        point2d<int> n1(p.row() + 2 * circle_r3_fast38s1[i][0],
                        p.col() + 2 * circle_r3_fast38s1[i][1]);
        if (frame_s1.has(n1))
          //distances[i+8] = (frame_s2(n1) - frame_s2(p));
          distances[i+8] = (frame_s2(n1) * 255);
        else
          distances[i+8] = 0.f;
      }
    }
    */
    {
      float min_diff = 9999999.f;
      float max_single_diff = 0.f;
      //int min_orientation = 0;
      pv = tex2D(s1_tex, p) / 255.f;

      for(int i = 0; i < 8; i++)
      {

        // float v1 = frame_s1(p.row() + circle_r3[i][0],
        //                     p.col() + circle_r3[i][1]);

        // float v2 = frame_s1(p.row() + circle_r3[(i+8)][0],
        //                     p.col() + circle_r3[(i+8)][1]);

        float v1 = tex2D(s1_tex,
                         p.col() + circle_r3[i][1],
                         p.row() + circle_r3[i][0]) / 255.f;

        float v2 = tex2D(s1_tex,
                         p.col() + circle_r3[(i+8)][1],
                         p.row() + circle_r3[(i+8)][0]) / 255.f;

        if (!(i % 2))
        {
            distances[i/2] = (v1 * 255.f);
            distances[i/2 + 4] = (v2 * 255.f);
        }

        {
          float diff = pv -
            (v1 + v2) / 2.f;
          float adiff = ::abs(diff);
          if (adiff < min_diff)
          {
            min_diff = adiff;
          }

          float sd = adiff;
          if (max_single_diff < sd) max_single_diff = sd;

        }
      }


      //pv = frame_s2(p);
      pv = tex2D(s2_tex, p) / 255.f;
      float min_diff_large = 9999999.f;
      //int min_orientation_large;
      for(int i = 0; i < 8; i++)
      {

        float v1 = tex2D(s2_tex,
                         p.col() + 2 * circle_r3[i][1],
                         p.row() + 2 * circle_r3[i][0]) / 255.f;

        float v2 = tex2D(s2_tex,
                         p.col() + 2 * circle_r3[(i+8)][1],
                         p.row() + 2 * circle_r3[(i+8)][0]) / 255.f;

        // float v1 = frame_s2(p.row() + 2 * circle_r3[i][0],
        //                     p.col() + 2 * circle_r3[i][1]);

        // float v2 = frame_s2(p.row() + 2 * circle_r3[(i+8)][0],
        //                     p.col() + 2 * circle_r3[(i+8)][1]);

        if (!(i % 2))
        {
            distances[8 + i/2] = (v1 * 255.f);
            distances[8 + i/2 + 4] = (v2 * 255.f);
        }

        {
          float diff = pv - (v1 + v2) / 2.f;
          float adiff = ::abs(diff);
          if (adiff < min_diff_large)
          {
            min_diff_large = adiff;
            if (min_diff_large < 0.01) break;
          }

        }
      }

      if (min_diff < min_diff_large)
      {
        min_diff = min_diff_large;
      }

      // if (min_diff < min_diff_large)
      // {
      //   min_diff = min_diff_large;
      //   int scale = 2;
      //   float minx = min_orientation_large;
      //   for (float delta = 0.5f; delta > 0.01f; delta /= 2.f)
      //   {
      //     {
      //       V v1 = r3_interpolation(frame_s2, p, minx - delta, scale);
      //       V v2 = r3_interpolation(frame_s2, p, minx - delta + 8, scale);
      //       //float adiff = ::abs(v1.x - v2.x);
      //       float adiff = ::abs(plj - (v1.x + v2.x) / 2.f);
      //       if (adiff < min_diff)
      //       {
      //         minx = minx - delta;
      //         min_diff = adiff;
      //       }
      //     }

      //     {
      //       V v1 = r3_interpolation(frame_s2, p, minx + delta, scale);
      //       V v2 = r3_interpolation(frame_s2, p, minx + delta + 8, scale);
      //       //float adiff = ::abs(v1.x - v2.x);
      //       float adiff = ::abs(plj - (v1.x + v2.x) / 2.f);
      //       if (adiff < min_diff)
      //       {
      //         minx = minx + delta;
      //         min_diff = adiff;
      //       }
      //     }
      //   }

      // }
      // else
      // Subpixel search.
      // {
      //   int scale = 1;
      //   float minx = min_orientation;
      //   for (float delta = 1.f; delta > 0.01f; delta /= 2.f)
      //   {
      //     {
      //       V v1 = r3_interpolation(frame_s1, p, minx - delta, scale);
      //       V v2 = r3_interpolation(frame_s1, p, minx - delta + 8, scale);
      //       //float adiff = ::abs(v1.x - v2.x);
      //       float adiff = ::abs(plj - (v1.x + v2.x) / 2.f);
      //       if (adiff < min_diff)
      //       {
      //         minx = minx - delta;
      //         min_diff = adiff;
      //       }
      //     }

      //     {
      //       V v1 = r3_interpolation(frame_s1, p, minx + delta, scale);
      //       V v2 = r3_interpolation(frame_s1, p, minx + delta + 8, scale);
      //       //float adiff = ::abs(v1.x - v2.x);
      //       float adiff = ::abs(plj - (v1.x + v2.x) / 2.f);
      //       if (adiff < min_diff)
      //       {
      //         minx = minx + delta;
      //         min_diff = adiff;
      //       }
      //     }
      //   }
      // }


      if (max_single_diff >= grad_thresh)
        min_diff = min_diff / max_single_diff;

      pertinence(p) = min_diff;
      out(p) = distances;

    }

  }

  template <typename T>
  __global__  void dfast382sl_to_color(kernel_image2d<dfast382sl> in,
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
  fast382sl_feature::fast382sl_feature(const domain_t& d)
    : gl_frame_(d),
      blurred_s1_(d),
      blurred_s2_(d),
      tmp1_(d),
      tmp2_(d),
      pertinence_(d),
      pertinence2_(d),
      f1_(d),
      f2_(d),
      fast382sl_color_(d),
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
  fast382sl_feature::update(const image2d<i_uchar1>& in)
  {
    swap_buffers();
    dim3 dimblock(32, 16, 1);
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    // local_jet_static_<0, 0, 1, 3>::run(in, blurred_s1_, tmp_);
    // local_jet_static_<0, 0, 2, 6>::run(in, blurred_s2_, tmp_);

    local_jet_static2_<0,0,1, 0,0,2, 6>::run(in, blurred_s1_, blurred_s2_, tmp1_, tmp2_);

    bindTexture2d(blurred_s1_, s1_tex);
    bindTexture2d(blurred_s2_, s2_tex);

    grad_thresh = Slider("grad_thresh").value() / 100.f;
    FAST382SL<i_uchar1><<<dimgrid, dimblock>>>
      (blurred_s1_, blurred_s2_, *f_, pertinence_, grad_thresh);

    // filter_pertinence<i_float1><<<dimgrid, dimblock>>>
    //   (pertinence_, pertinence2_);
    // copy(pertinence2_, pertinence_);

    cudaUnbindTexture(s1_tex);
    cudaUnbindTexture(s2_tex);
    check_cuda_error();
  }

  inline
  const image2d<i_float4>&
  fast382sl_feature::feature_color() const
  {
    return fast382sl_color_;
  }

  inline
  void
  fast382sl_feature::swap_buffers()
  {
    std::swap(f_prev_, f_);
  }

  inline
  const fast382sl_feature::domain_t&
  fast382sl_feature::domain() const
  {
    return f1_.domain();
  }

  inline
  image2d<dfast382sl>&
  fast382sl_feature::previous_frame()
  {
    return *f_prev_;
  }

  inline
  image2d<dfast382sl>&
  fast382sl_feature::current_frame()
  {
    return *f_;
  }

  inline
  image2d<i_float1>&
  fast382sl_feature::pertinence()
  {
    return pertinence_;
  }

  inline
  kernel_fast382sl_feature::kernel_fast382sl_feature(fast382sl_feature& f)
    : pertinence_(f.pertinence()),
      //f_prev_(f.previous_frame()),
      f_(f.current_frame())
  {
  }


  // inline
  // __device__ float
  // kernel_fast382sl_feature::distance(const point2d<int>& p_prev,
  //                               const point2d<int>& p_cur)
  // {
  //   return cuimg::distance_mean(f_prev_(p_prev), f_(p_cur));
  // }

  // inline
  // __device__ float
  // kernel_fast382sl_feature::distance(const dfast382sl& a,
  //                               const dfast382sl& b)
  // {
  //   return cuimg::distance_mean(a, b);
  // }


  inline
  __device__ float
  kernel_fast382sl_feature::distance_linear_s2(const dfast382sl& a,
                                const dfast382sl& b)
  {
    return cuimg::distance_mean_linear_s2(a, b);
  }

  inline
  __device__ float
  kernel_fast382sl_feature::distance_linear(const dfast382sl& a,
                                const dfast382sl& b)
  {
    return cuimg::distance_mean_linear(a, b);
  }

  inline
  __device__ float
  kernel_fast382sl_feature::distance_s2(const dfast382sl& a,
                                const dfast382sl& b)
  {
    return cuimg::distance_mean_s2(a, b);
  }

  // inline __device__
  // kernel_image2d<dfast382sl>&
  // kernel_fast382sl_feature::previous_frame()
  // {
  //   return f_prev_;
  // }

  inline __device__
  kernel_image2d<dfast382sl>&
  kernel_fast382sl_feature::current_frame()
  {
    return f_;
  }

  inline __device__
  kernel_image2d<i_float1>&
  kernel_fast382sl_feature::pertinence()
  {
    return pertinence_;
  }

    inline
  void
  fast382sl_feature::display() const
  {
#ifdef WITH_DISPLAY
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(pertinence_.domain(), dimblock);

    dfast382sl_to_color<int><<<dimgrid, dimblock>>>(*f_, fast382sl_color_);
    ImageView("test") <<= dg::dl() - gl_frame_ - blurred_s1_ - blurred_s2_ - pertinence_ - fast382sl_color_;
#endif
  }
}

#endif // ! CUIMG_FAST382SL_FEATURE_HPP_
