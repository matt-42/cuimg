#ifndef CUIMG_FAST382S_FEATURE_HPP_
# define CUIMG_FAST382S_FEATURE_HPP_

# include <cuda_runtime.h>
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

# include <cuimg/dige.h>

#include <dige/widgets/image_view.h>

using dg::dl;
using namespace dg::widgets;

namespace cuimg
{


  // inline
  // __host__ __device__
  // float distance_mean(const dfast382s& a, const dfast382s& b)
  // {
  //   if (::abs(a.pertinence - b.pertinence) > 0.2f ||
  //       b.pertinence < 0.1f ||
  //       a.pertinence < 0.1f
  //       )
  //     return 99999.f;
  //   else
  //   {
  //     float d = 0;
  //     for (unsigned i = 0; i < 16; i++)
  //     {
  //       float tmp = (a.distances[i] - b.distances[i]);
  //       d += tmp * tmp;
  //     }

  //     return ::sqrt(float(d)) / ::sqrt(16.f);
  //   }
  // }


  inline
  __host__ __device__
  float distance_mean_linear(const dfast382s& a, const dfast382s& b)
  {
    float d = 0;
    for (unsigned i = 0; i < 16; i++)
      d += ::abs(a.distances[i] - b.distances[i]);

    return float(d) / 16.f;
  }


  // inline
  // __host__ __device__
  // float distance_mean_linear(const dfast382s& a, const dfast382s& b)
  // {
  //   float d = 0;
  //   for (unsigned i = 0; i < 16; i++)
  //   {
  //     float tmp = (a.distances[i] - b.distances[i]);
  //     d += tmp * tmp;
  //   }

  //   return ::sqrt(float(d)) / ::sqrt(16.f);
  // }



  inline
  __host__ __device__
  float distance_mean_linear_s2(const dfast382s& a, const dfast382s& b)
  {
    float d = 0;
    for (unsigned i = 8; i < 16; i++)
    {
      float tmp = (a.distances[i] - b.distances[i]);
      d += tmp * tmp;
    }

    return ::sqrt(float(d)) / ::sqrt(8.f);
  }

  inline
  __host__ __device__
  float distance_max_linear(const dfast382s& a, const dfast382s& b)
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
      float d = 0;
      for (unsigned i = 8; i < 16; i++)
      {
        float tmp = (a.distances[i] - b.distances[i]);
        d += tmp * tmp;
      }

      return ::sqrt(float(d)) / ::sqrt(8.f);
    }
  }


  // inline
  // __host__ __device__
  // float distance_min(const dfast382s& a, const dfast382s& b)
  // {
  //   if (::abs(a.pertinence - b.pertinence) > 0.1f ||
  //       b.pertinence < 0.15f ||
  //       a.pertinence < 0.15f)
  //     return 99999.f;
  //   else
  //   {
  //     float d = 9999999.f;
  //     for (unsigned i = 0; i < 16; i++)
  //     {
  //       float tmp = ::abs(a.distances[i] - b.distances[i]);
  //       if (tmp < d)
  //         d = tmp;
  //     }

  //     return d;
  //   }
  // }

  // inline
  // __host__ __device__
  // float distance_max(const dfast382s& a, const dfast382s& b)
  // {
  //   if (::abs(a.pertinence - b.pertinence) > 0.1f ||
  //       b.pertinence < 0.15f ||
  //       a.pertinence < 0.15f)
  //     return 99999.f;
  //   else
  //   {
  //     float d = 0.f;
  //     for (unsigned i = 0; i < 16; i++)
  //     {
  //       float tmp = ::abs(a.distances[i] - b.distances[i]);
  //       if (tmp > d)
  //         d = tmp;
  //     }

  //     return d;
  //   }
  // }

  __host__ __device__ inline
  dfast382s operator+(const dfast382s& a, const dfast382s& b)
  {
    dfast382s res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] + b.distances[i];
    //res.pertinence = a.pertinence + b.pertinence;
    return res;
  }

  __host__ __device__ inline
  dfast382s operator-(const dfast382s& a, const dfast382s& b)
  {
    dfast382s res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] - b.distances[i];
    //res.pertinence = a.pertinence - b.pertinence;
    return res;
  }

  template <typename S>
  __host__ __device__ inline
  dfast382s operator/(const dfast382s& a, const S& s)
  {
    dfast382s res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] / s;
    //res.pertinence = a.pertinence / s;
    return res;
  }

  template <typename S>
  __host__ __device__ inline
  dfast382s operator*(const dfast382s& a, const S& s)
  {
    dfast382s res;
    for (unsigned i = 0; i < 16; i++)
      res.distances[i] = a.distances[i] * s;
    //res.pertinence = a.pertinence * s;
    return res;
  }

  __host__ __device__ inline
  dfast382s weighted_mean(const dfast382s& a, float aw, const dfast382s& b, float bw)
  {
    dfast382s res;
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
  __constant__ const int circle_r3_fast382s[8][2] = {
    {-1, 0}, {-1, 1},
    { 0, 1}, { 1, 1},
    { 1, 0}, { 1,-1},
    { 0,-1}, {-1,-1}
  };
  */


  template <typename V>
  inline
  __device__ V r3_interpolation(kernel_image2d<V> in,
                                point2d<int> p,
                                float x,
                                int scale)
  {
    if (x < 0) x += 16.f;
    if (x >= 16) x -= 16.f;

    point2d<int> p1(p.row() + scale * circle_r3[int(x)][0],
                    p.col() + scale * circle_r3[int(x)][1]);

    int x2 = int(ceil(x)) % 16;
    point2d<int> p2(p.row() + scale * circle_r3[int(x2)][0],
                    p.col() + scale * circle_r3[int(x2)][1]);

    if (in.has(p1) && in.has(p2))
    {
      V v1 = in(p1);

      V v2 = in(p2);

      return (v1 * (1 - (x2 - x)) + v2 * (x2 - x));
    }
    else
      return zero();
  }



  template <typename V>
  __global__ void filter_pertinence(kernel_image2d<i_float1> pertinence,
                                    kernel_image2d<i_float1> out)
  {
    point2d<int> p = thread_pos2d();
    if (!pertinence.has(p))
      return;

    float max = pertinence(p).x;
    for(unsigned i = 0; i < 25; i++)
    {
      point2d<int> n(p.row() + c25[i][0],
                     p.col() + c25[i][1]);
      if (pertinence.has(n) && max < pertinence(n).x)
        max = pertinence(n).x;
    }

    if (max > 0.3f)
      out(p) = pertinence(p).x;
    else
      out(p) = 0.f;
  }

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
          //distances[i] = (frame_s1(n1) - frame_s1(p));
          distances[i] = (frame_s1(n1));
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
          distances[i+8] = (frame_s2(n1));
        else
          distances[i+8] = 0.f;
      }
    }

    {
      float min_diff = 9999999.f;
      float max_diff = 0.f;
      float max_single_diff = 0.f;
      int min_orientation = 0;
      bool sign = false;

      float diff1 = 0.f;
      float diff2 = 0.f;

      bool is_flat = false;
      for(unsigned i = 0; i < 8; i++)
      {
        point2d<int> n1(p.row() + circle_r3[i][0],
                        p.col() + circle_r3[i][1]);
        point2d<int> n2(p.row() + circle_r3[(i+8)][0],
                        p.col() + circle_r3[(i+8)][1]);

        if (frame_s1.has(n1) && frame_s1.has(n2))
        {
          float diff = frame_s1(p).x - (frame_s1(n1).x + frame_s1(n2).x) / 2.f;
          //float diff = ::abs(plj - frame_s1(n1).x) + ::abs(plj - frame_s1(n2).x);
          //float diff = (for_pertinence(n1).x - for_pertinence(n2).x);
          float adiff = ::abs(diff);
          if (adiff < min_diff)
          {
            min_diff = adiff;
            min_orientation = i;
            // if (i && sign != diff > 0)
            //   is_flat = true;
            // else
            //   sign = diff > 0;

          }

          float sd = adiff;
          //float sd = ::abs(frame_s1(n1).x - frame_s1(n2).x);

          //float sd = ::max(::abs(plj - frame_s1(n1).x), ::abs(plj - frame_s1(n2).x));

          //sd = ::max(sd, ::abs(frame_s1(n1).x - frame_s1(n2).x));
          if (max_single_diff < sd) max_single_diff = sd;

        }
      }

      //max_single_diff = (frame_s1(p) + frame_s2(p));

      // float ad2 = 9999.f;
      // unsigned perp = (min_orientation+8+4)%8;
      // for (unsigned i = 0; i < 10; i += 8)
      // {
      //   point2d<int> n(p.row() + circle_r3[perp + i][0],
      //                  p.col() + circle_r3[perp + i][1]);
      //   // point2d<int> n(p.row() + circle_r3[min_orientation + i][0],
      //   //                p.col() + circle_r3[min_orientation + i][1]);

      //   point2d<int> n1(n.row() + circle_r3[min_orientation][0],
      //                   n.col() + circle_r3[min_orientation][1]);
      //   point2d<int> n2(n.row() + circle_r3[(min_orientation+8)][0],
      //                   n.col() + circle_r3[(min_orientation+8)][1]);

      //   if (frame_s1.has(n1) && frame_s1.has(n2))
      //   {
      //     float diff = frame_s1(n) - (frame_s1(n1).x + frame_s1(n2).x) / 2.f;
      //     float adiff = ::abs(diff);

      //     //if (ad2 > adiff) ad2 = adiff;
      //     //ad2 += adiff;
      //     if (adiff > min_diff)
      //     {
      //       min_diff = adiff;
      //     }
      //   }
      // }
      // ad2 /= 2.f;

      //min_diff = ad2;
      // {
      //   point2d<int> n1(p.row() + 1.5f * circle_r3[min_orientation][0],
      //                   p.col() + 1.5f * circle_r3[min_orientation][1]);
      //   point2d<int> n2(p.row() + 1.5f * circle_r3[(min_orientation+8)][0],
      //                   p.col() + 1.5f * circle_r3[(min_orientation+8)][1]);
      //   if (frame_s1.has(n1) && frame_s1.has(n2))
      //   {
      //     float diff = frame_s1(p) - (frame_s1(n1).x + frame_s1(n2).x) / 2.f;
      //     float adiff = ::abs(diff);
      //     //if (adiff > min_diff)
      //     {
      //       min_diff = adiff;
      //     }
      //   }
      // }

      // {
      //   point2d<int> n1(p.row() + 1.5f * circle_r3[min_orientation][0],
      //                   p.col() + 1.5f * circle_r3[min_orientation][1]);
      //   point2d<int> n2(p.row() + 1.5f * circle_r3[(min_orientation+8)][0],
      //                   p.col() + 1.5f * circle_r3[(min_orientation+8)][1]);
      //   if (frame_s1.has(n1) && frame_s1.has(n2))
      //   {
      //     float diff = frame_s2(n1) - (frame_s2(n1).x + frame_s2(n2).x) / 2.f;
      //     float adiff = ::abs(diff);
      //     if (adiff < min_diff)
      //     {
      //       min_diff = adiff;
      //     }
      //   }
      // }


      if (is_flat)
        min_diff = 0.f;

      float min_diff_large = 9999999.f;
      int min_orientation_large;
      is_flat = false;
      for(unsigned i = 0; i < 8; i++)
      {
        point2d<int> n1(p.row() + 2 * circle_r3[i][0],
                        p.col() + 2 * circle_r3[i][1]);
        point2d<int> n2(p.row() + 2 * circle_r3[(i+8)][0],
                        p.col() + 2 * circle_r3[(i+8)][1]);

        if (frame_s1.has(n1) && frame_s1.has(n2))
        {
          float diff = frame_s2(p) - (frame_s2(n1).x + frame_s2(n2).x) / 2.f;
          //float diff = (for_pertinence(n1).x - for_pertinence(n2).x);
          float adiff = ::abs(diff);
          if (adiff < min_diff_large)
          {
            min_diff_large = adiff;
            min_orientation_large = i;
            // if (i && sign != (diff > 0))
            //   is_flat = true;
            // else
            //   sign = diff > 0;
          }
          float sd = adiff;

          //          float sd = ::max(::abs(plj - frame_s1(n1).x), ::abs(plj - frame_s1(n2).x));

          // if (max_single_diff < sd) max_single_diff = sd;

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

      for(unsigned i = 0; i < 16; i++)
         out(p).distances[i] = float(distances[i]);

  /*    for(unsigned i = 0; i < 16; i++)
        out(p).distances[i] = distances[i] / max_single_diff;

      for(unsigned i = 1; i < 7; i++)
        out(p).distances[i] = (distances[i] + 0.5f * (distances[i+1] + distances[i-1])) / (2.f * max_single_diff);

      for(unsigned i = 9; i < 15; i++)
        out(p).distances[i] = (distances[i] + 0.5f * (distances[i+1] + distances[i-1])) / (2.f * max_single_diff);

      out(p).distances[0] = (distances[0] + 0.5f * (distances[1] + distances[7])) / (2.f * max_single_diff);
      out(p).distances[7] = (distances[7] + 0.5f * (distances[0] + distances[6])) / (2.f * max_single_diff);
      out(p).distances[8] = (distances[8] + 0.5f * (distances[9] + distances[15])) / (2.f * max_single_diff);
      out(p).distances[15] = (distances[15] + 0.5f * (distances[8] + distances[14])) / (2.f * max_single_diff);
      */
      if (max_single_diff >= grad_thresh)
      {

        pertinence(p) = min_diff / max_single_diff;

        //min(min_diff / max_single_diff, max_single_diff);

        //out(p).pertinence = 1.f;
        //pertinence(p) = 1.f;

      }
      else
      {
        pertinence(p) = min_diff;
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
    for (unsigned i = 0; i < 8; i+=2)
      //res[i/2] = (::abs(in(p).distances[i]) + ::abs(in(p).distances[i+1])) / (2*127.f);
      res[i/2] = (::abs(in(p).distances[i]) + ::abs(in(p).distances[i+1])) / 2;
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
      pertinence2_(d),
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
    update(gl_frame_);
  }

  inline
  void
  fast382s_feature::update(const image2d<i_float1>& in)
  {
    swap_buffers();
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    local_jet_static_<0, 0, 1, 10>::run(in, blurred_s1_, tmp_);
    local_jet_static_<0, 0, 2, 15>::run(in, blurred_s2_, tmp_);

    grad_thresh = Slider("grad_thresh").value() / 100.f;
    FAST382S<i_float1><<<dimgrid, dimblock>>>
      (color_blurred_, blurred_s1_, blurred_s2_, *f_, pertinence_, grad_thresh);

    filter_pertinence<i_float1><<<dimgrid, dimblock>>>
      (pertinence_, pertinence2_);
    copy(pertinence2_, pertinence_);

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


  // inline
  // __device__ float
  // kernel_fast382s_feature::distance(const point2d<int>& p_prev,
  //                               const point2d<int>& p_cur)
  // {
  //   return cuimg::distance_mean(f_prev_(p_prev), f_(p_cur));
  // }

  // inline
  // __device__ float
  // kernel_fast382s_feature::distance(const dfast382s& a,
  //                               const dfast382s& b)
  // {
  //   return cuimg::distance_mean(a, b);
  // }


  inline
  __device__ float
  kernel_fast382s_feature::distance_linear_s2(const dfast382s& a,
                                const dfast382s& b)
  {
    return cuimg::distance_mean_linear_s2(a, b);
  }

  inline
  __device__ float
  kernel_fast382s_feature::distance_linear(const dfast382s& a,
                                const dfast382s& b)
  {
    return cuimg::distance_mean_linear(a, b);
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

    inline
  void
  fast382s_feature::display() const
  {
#ifdef WITH_DISPLAY
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(pertinence_.domain(), dimblock);

    dfast382s_to_color<int><<<dimgrid, dimblock>>>(*f_, fast382s_color_);
    ImageView("test") <<= dg::dl() - gl_frame_ - blurred_s1_ - blurred_s2_ - pertinence_ - fast382s_color_;
#endif
  }
}

#endif // ! CUIMG_FAST382S_FEATURE_HPP_
