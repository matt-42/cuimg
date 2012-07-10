#ifndef CUIMG_FAST382SL_FEATURE_HPP_
# define CUIMG_FAST382SL_FEATURE_HPP_

# include <cuimg/gpu/cuda.h>

# include <cmath>

# include <cuimg/target.h>
# include <cuimg/copy.h>
# include <cuimg/pw_call.h>
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

# ifdef WITH_DISPLAY
#  include <cuimg/dige.h>
#  include <dige/widgets/image_view.h>
# endif

using dg::dl;
using namespace dg::widgets;

namespace cuimg
{

  #define s1_tex UNIT_STATIC(hawzxb)
  #define s2_tex UNIT_STATIC(xyhgtk)

#ifdef NVCC
  texture<float1, cudaTextureType2D, cudaReadModeElementType> s1_tex;
  texture<float1, cudaTextureType2D, cudaReadModeElementType> s2_tex;
#else
  int s1_tex;
  int s2_tex;
#endif

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


#define FAST382SL_sig(T, V)                      \
  kernel_image2d<i_float4>,                     \
  kernel_image2d<V>,                            \
    kernel_image2d<V>,                          \
    kernel_image2d<i_float1>,                   \
    float,                                      \
    &FAST382SL<V>

  template <typename V>
  __device__ void FAST382SL(thread_info<GPU> ti,
                            kernel_image2d<i_float4> frame_color,
                            kernel_image2d<V> frame_s1,
                            kernel_image2d<V> frame_s2,
                            kernel_image2d<i_float1> pertinence,
                            float grad_thresh)
  {
    point2d<int> p = thread_pos2d(ti);
    if (!frame_s1.has(p))//; || track(p).x == 0)
      return;


    if (p.row() < 6 || p.row() >= pertinence.domain().nrows() - 6 ||
        p.col() < 6 || p.col() >= pertinence.domain().ncols() - 6)
    {
      pertinence(p).x = 0.f;
      return;
    }

    float pv;

    {
      float min_diff = 9999999.f;
      float max_single_diff = 0.f;
      pv = tex2D(flag<GPU>(), s1_tex, frame_s1, p).x;
      for(int i = 0; i < 8; i++)
      {

        float v1 = tex2D(flag<GPU>(), s1_tex, frame_s1,
                                 p.col() + circle_r3[i][1],
                                 p.row() + circle_r3[i][0]).x;

        float v2 = tex2D(flag<GPU>(), s1_tex, frame_s1,
                                 p.col() + circle_r3[(i+8)][1],
                                 p.row() + circle_r3[(i+8)][0]).x;

        {
          float diff = pv -
            (v1 + v2) / 2.f;


          float adiff = fabs(diff);

          if (adiff < min_diff)
	  {
            min_diff = adiff;

            /*	    if (min_diff < 0.01)
	    {
	      min_diff = 0;
	      break;
              }*/
	  }

          float contrast = std::max(fabs(pv - v1), fabs(pv - v2));
          if (max_single_diff < contrast) max_single_diff = contrast;
        }
      }
      /*
      pv = tex2D(flag<GPU>(), s2_tex, frame_s2, p).x;
      float min_diff_large = 9999999.f;
      float max_single_diff_large = 0.f;
      //int min_orientation_large;
      for(int i = 0; i < 8; i++)
      {

        float v1 = tex2D(flag<GPU>(), s2_tex, frame_s2,
                         p.col() + 2 * circle_r3[i][1],
                         p.row() + 2 * circle_r3[i][0]).x;

        float v2 = tex2D(flag<GPU>(), s2_tex, frame_s2,
                         p.col() + 2 * circle_r3[(i+8)][1],
                         p.row() + 2 * circle_r3[(i+8)][0]).x;

        {
          float diff = pv - (v1 + v2) / 2.f;
          float adiff = fabs(diff);

          if (adiff < min_diff_large)
          {
            min_diff_large = adiff;
	    if (min_diff_large < 0.01)
	    {
	      min_diff_large = 0;
	      break;
	    }
          }

          float contrast = max(fabs(pv - v1), fabs(pv - v2));
          if (max_single_diff < contrast) max_single_diff = contrast;
          //if (max_single_diff_large < adiff) max_single_diff_large = adiff;
        }

      }


      if (min_diff < min_diff_large)
      {
        min_diff = min_diff_large;
        max_single_diff = max_single_diff_large;
      }
      */

      if (max_single_diff >= grad_thresh)
      {
        min_diff = min_diff / max_single_diff;
      }
      else
        min_diff = 0;

      pertinence(p) = min_diff;

    }

  }

  template <typename V>
  void FAST382SL(thread_info<CPU> ti,
                 kernel_image2d<i_float4> frame_color,
                 kernel_image2d<V> frame_s1,
                 kernel_image2d<V> frame_s2,
                 // kernel_image2d<dfast382sl> out,
                 kernel_image2d<i_float1> pertinence,
                 float grad_thresh)
  {

    point2d<int> p = thread_pos2d(ti);
    if (!frame_s1.has(p))//; || track(p).x == 0)
      return;


    if (p.row() < 6 || p.row() >= pertinence.domain().nrows() - 6 ||
        p.col() < 6 || p.col() >= pertinence.domain().ncols() - 6)
    {
      pertinence(p).x = 0.f;
      return;
    }

    // dfast382sl distances;

    float pv;

    {
      float min_diff = 9999999.f;
      float max_single_diff = 0.f;
      pv = tex2D(flag<CPU>(), s1_tex, frame_s1, p).x;
      for(int i = 0; i < 8; i++)
      {

        float v1 = tex2D(flag<CPU>(), s1_tex, frame_s1,
                                 p.col() + circle_r3_h[i][1],
                                 p.row() + circle_r3_h[i][0]).x;

        float v2 = tex2D(flag<CPU>(), s1_tex, frame_s1,
                                 p.col() + circle_r3_h[(i+8)][1],
                                 p.row() + circle_r3_h[(i+8)][0]).x;

        {
          float diff = pv -
            (v1 + v2) / 2.f;


          float adiff = fabs(diff);

          if (adiff < min_diff)
            min_diff = adiff;



          if (max_single_diff < adiff) max_single_diff = adiff;

        }
      }

      pv = tex2D(flag<CPU>(), s2_tex, frame_s2, p).x;
      float min_diff_large = 9999999.f;
      float max_single_diff_large = 0.f;
      //int min_orientation_large;
      for(int i = 0; i < 8; i++)
      {

        float v1 = tex2D(flag<CPU>(), s2_tex, frame_s2,
                         p.col() + 2 * circle_r3_h[i][1],
                         p.row() + 2 * circle_r3_h[i][0]).x;

        float v2 = tex2D(flag<CPU>(), s2_tex, frame_s2,
                         p.col() + 2 * circle_r3_h[(i+8)][1],
                         p.row() + 2 * circle_r3_h[(i+8)][0]).x;

        // if (!(i % 2))
        // {
        //     distances[8 + i/2] = (v1 * 255);
        //     distances[8 + i/2 + 4] = (v2 * 255);
        // }

        {
          float diff = pv - (v1 + v2) / 2.f;
          float adiff = fabs(diff);

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
      // out(p) = distances;

    }

  }

#define dfast382sl_to_color_sig(T)              \
kernel_image2d<dfast382sl> in,                  \
    kernel_image2d<i_float4> out,               \
    &dfast382sl_to_color<T>

  template <target target>
  __host__ __device__  void dfast382sl_to_color(thread_info<target> ti,
                                                kernel_image2d<dfast382sl> in,
                                                kernel_image2d<i_float4> out)
  {
    point2d<int> p = thread_pos2d(ti);
    if (!in.has(p))
      return;

    i_float4 res;
    for (unsigned i = 0; i < 8; i+=2)
      //res[i/2] = (::abs(in(p).distances[i]) + ::abs(in(p).distances[i+1])) / (2*127.f);
      res[i/2] = (::abs(float(in(p).distances[i]) / 255.f) + ::abs(float(in(p).distances[i+1]) / 255.f)) / 2;
    res.w = 1.f;
    out(p) = res;
  }

  template <target T>
  inline
  fast382sl_feature<T>::fast382sl_feature(const domain_t& d)
    : gl_frame_(d),
      blurred_s1_(d),
      blurred_s2_(d),
      tmp_(d),
      pertinence_(d),
      pertinence2_(d),
      f1_(d),
      f2_(d),
      fast382sl_color_(d),
      color_blurred_(d),
      color_tmp_(d),
      grad_thresh(0.02f),
      frame_cpt_(0)
  {
    f_prev_ = &f1_;
    f_ = &f2_;
#ifndef NO_CUDA
    cudaStreamCreate(&cuda_stream_);
#endif

  }

  template <target T>
  inline
  void
  fast382sl_feature<T>::update(const image2d_f1& in, const image2d_f1& in_s2)
  {
    SCOPE_PROF(fast382sl_feature_update);

    frame_cpt_++;
    swap_buffers();
    dim3 dimblock(16, 16, 1);
    if (T == CPU)
      dimblock = dim3(in.ncols(), 1, 1);

    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    local_jet_static_<0, 0, 1, 1>::run(in, blurred_s1_, tmp_, 0, dimblock);
    local_jet_static_<0, 0, 2, 2>::run(in, blurred_s2_, tmp_, 0, dimblock);

    //local_jet_static2_<0,0,1, 0,0,2, 6>::run(in, blurred_s1_, blurred_s2_, tmp_, pertinence2_);

    //if (!(frame_cpt_ % 5))
    {
    if (target == GPU)
    {
      bindTexture2d(blurred_s1_, s1_tex);
      bindTexture2d(blurred_s2_, s2_tex);
    }

    // grad_thresh = Slider("grad_thresh").value() / 100.f;
    START_PROF(kernel);
    pw_call<FAST382SL_sig(target, i_float1)>(flag<target>(), dimgrid, dimblock,
                                             color_blurred_, blurred_s1_, blurred_s2_,
                                             //*f_,
					     pertinence_, grad_thresh);
    END_PROF(kernel);

    // filter_pertinence<i_float1><<<dimgrid, dimblock>>>
    //   (pertinence_, pertinence2_);
    // copy(pertinence2_, pertinence_);

    if (target == GPU)
    {
      cudaUnbindTexture(s1_tex);
      cudaUnbindTexture(s2_tex);
      check_cuda_error();
    }
    }
  }

  template <target T>
  inline
  const typename fast382sl_feature<T>::image2d_f4&
  fast382sl_feature<T>::feature_color() const
  {
    return fast382sl_color_;
  }

  template <target T>
  inline
  void
  fast382sl_feature<T>::swap_buffers()
  {
    std::swap(f_prev_, f_);
  }

  template <target T>
  inline
  const typename fast382sl_feature<T>::domain_t&
  fast382sl_feature<T>::domain() const
  {
    return f1_.domain();
  }

  template <target T>
  inline
  typename fast382sl_feature<T>::image2d_D&
  fast382sl_feature<T>::previous_frame()
  {
    return *f_prev_;
  }


  template <target T>
  inline
  typename fast382sl_feature<T>::image2d_D&
  fast382sl_feature<T>::current_frame()
  {
    return *f_;
  }

  template <target T>
  inline
  typename fast382sl_feature<T>::image2d_f1&
  fast382sl_feature<T>::s1()
  {
    return blurred_s1_;
  }

  template <target T>
  inline
  typename fast382sl_feature<T>::image2d_f1&
  fast382sl_feature<T>::s2()
  {
    return blurred_s2_;
  }

  template <target T>
  inline
  typename fast382sl_feature<T>::image2d_f1&
  fast382sl_feature<T>::pertinence()
  {
    return pertinence_;
  }


  template <target T>
  inline
  const typename fast382sl_feature<T>::image2d_f1&
  fast382sl_feature<T>::pertinence() const
  {
    return pertinence_;
  }


    template <target target>
    __host__ __device__
    inline
    kernel_fast382sl_feature::kernel_fast382sl_feature(fast382sl_feature<target>& f)
    : pertinence_(f.pertinence()),
      f_prev_(f.previous_frame()),
      // f_(f.current_frame()),
      s1_(f.s1()),
      s2_(f.s2())
  {
    for (unsigned i = 0; i < 16; i++)
    {
      point2d<int> p(10,10);
      offsets_s1[i] = (long(&s1_(p + i_int2(circle_r3[i]))) - long(&s1_(p))) / sizeof(i_float1);
      offsets_s2[i] = (long(&s2_(p + i_int2(circle_r3[i])*2)) - long(&s2_(p))) / sizeof(i_float1);
    }
  }


  // inline
  // __host__ __device__ float
  // kernel_fast382sl_feature<T>::distance(const point2d<int>& p_prev,
  //                               const point2d<int>& p_cur)
  // {
  //   return cuimg::distance_mean(f_prev_(p_prev), f_(p_cur));
  // }

  // inline
  // __host__ __device__ float
  // kernel_fast382sl_feature<T>::distance(const dfast382sl& a,
  //                               const dfast382sl& b)
  // {
  //   return cuimg::distance_mean(a, b);
  // }


  inline
  __host__ __device__ float
  kernel_fast382sl_feature::distance_linear_s2(const dfast382sl& a,
                                const dfast382sl& b)
  {
    return cuimg::distance_mean_linear_s2(a, b);
  }


  // inline
  // __host__ __device__ float
  // kernel_fast382sl_feature::distance_linear(const point2d<int>& p_prev,
  //                                           const point2d<int>& p_cur)
  // {
  //   return cuimg::distance_mean_linear(f_(p_prev), f_(p_cur));
  // }

  inline
  __host__ __device__ float
  kernel_fast382sl_feature::distance_linear(const dfast382sl& a,
                                const dfast382sl& b)
  {
    return cuimg::distance_mean_linear(a, b);
  }


  inline
  __host__ __device__ dfast382sl
  kernel_fast382sl_feature::weighted_mean(const dfast382sl& a, float aw,
                                          const point2d<int>& n, float bw)
  {
    return new_state(n);
  }


  inline
  __host__ __device__ float kernel_fast382sl_feature::distance_linear(const dfast382sl& a,
                                                                      const point2d<int>& n)
  {
    float d = 0.f;

    i_float1* data = &s1_(n);
    for(int i = 0; i < 8; i ++)
    {
      float v = data[offsets_s1[i*2]].x * 255.f;
      d += fabs(v - a[i]);
    }

    data = &s2_(n);
    for(int i = 0; i < 8; i ++)
    {
      float v = data[offsets_s2[i*2]].x * 255.f;
      d += fabs(v - a[8+i]);
    }

    return d / (255.f * 16.f);
  }

  inline
  __host__ __device__ float
  kernel_fast382sl_feature::distance_s2(const dfast382sl& a,
					const dfast382sl& b)
  {
    return cuimg::distance_mean_s2(a, b);
  }

  // inline __host__ __device__
  // kernel_image2d<dfast382sl>&
  // kernel_fast382sl_feature::previous_frame()
  // {
  //   return f_prev_;
  // }

  // inline __host__ __device__
  // kernel_image2d<dfast382sl>&
  // kernel_fast382sl_feature::current_frame()
  // {
  //   return f_;
  // }


  inline __host__ __device__
  kernel_image2d<i_float1>&
  kernel_fast382sl_feature::s1()
  {
    return s1_;
  }

  inline __host__ __device__
  kernel_image2d<i_float1>&
  kernel_fast382sl_feature::s2()
  {
    return s2_;
  }

  inline __host__ __device__
  kernel_image2d<i_float1>&
  kernel_fast382sl_feature::pertinence()
  {
    return pertinence_;
  }

  inline
  __host__ __device__ dfast382sl
  kernel_fast382sl_feature::new_state(const point2d<int>& n)
  {
    dfast382sl b;
    for(int i = 0; i < 16; i += 2)
      b[i/2] = s1_(n.row() + circle_r3[i][0],
    		   n.col() + circle_r3[i][1]).x * 255.f;
    for(int i = 0; i < 16; i += 2)
      b[i/2+8] = s2_(n.row() + 2 * circle_r3[i][0],
    		     n.col() + 2 * circle_r3[i][1]).x * 255.f;
    return b;
    //return f_(n);
  }

  template <target T>
  inline
  void
  fast382sl_feature<T>::display() const
  {
#ifdef WITH_DISPLAY
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(pertinence_.domain(), dimblock);

    // pw_call<dfast382sl_to_color_sig(target) >(flag<target>(), dimgrid, dimblock, *f_, fast382sl_color_);

    ImageView("test") <<= dg::dl() - gl_frame_ - blurred_s1_ - blurred_s2_ - pertinence_ - fast382sl_color_;
#endif
  }
}

#endif // ! CUIMG_FAST382SL_FEATURE_HPP_
