#ifndef CUIMG_FAST382S_FEATURE_H_
# define CUIMG_FAST382S_FEATURE_H_

# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{

  struct dfast382s
  {
    __host__ __device__ dfast382s() {}

    __host__ __device__ dfast382s(const dfast382s& o)
    {
      pertinence = o.pertinence;
      for (unsigned i = 0; i < 16; i++) distances[i] = o.distances[i];
    }

    __host__ __device__
    dfast382s& operator=(const dfast382s& o)
    {
      pertinence = o.pertinence;
      for (unsigned i = 0; i < 16; i++) distances[i] = o.distances[i];
      return *this;
    }

    float pertinence;
    float distances[16];
  };

  __host__ __device__ inline
  dfast382s operator+(const dfast382s& a, const dfast382s& b);
  __host__ __device__ inline
  dfast382s operator-(const dfast382s& a, const dfast382s& b);

  template <typename S>
  __host__ __device__ inline
  dfast382s operator/(const dfast382s& a, const S& s);

  template <typename S>
  __host__ __device__ inline
  dfast382s operator*(const dfast382s& a, const S& s);

  class kernel_fast382s_feature;

  class fast382s_feature
  {
  public:
    typedef dfast382s feature_t;
    typedef obox2d<point2d<int> > domain_t;

    typedef kernel_fast382s_feature kernel_type;

    inline fast382s_feature(const domain_t& d);

    inline void update(const image2d<i_float4>& in);

    inline const domain_t& domain() const;

    inline image2d<dfast382s>& previous_frame();
    inline image2d<dfast382s>& current_frame();
    inline image2d<i_float1>& pertinence();

    const image2d<i_float4>& feature_color() const;


  private:
    inline void swap_buffers();


    image2d<i_float1> gl_frame_;
    image2d<i_float1> blurred_s1_;
    image2d<i_float1> blurred_s2_;
    image2d<i_float1> tmp_;

    image2d<i_float1> pertinence_;

    image2d<dfast382s> f1_;
    image2d<dfast382s> f2_;

    image2d<dfast382s>* f_prev_;
    image2d<dfast382s>* f_;

    image2d<i_float4> fast382s_color_;

    image2d<i_float4> color_blurred_;
    image2d<i_float4> color_tmp_;

    float grad_thresh;
  };

  class kernel_fast382s_feature
  {
  public:
    typedef dfast382s feature_t;

    inline kernel_fast382s_feature(fast382s_feature& f);


    inline
    __device__ float distance(const point2d<int>& p_prev,
                              const point2d<int>& p_cur);

    inline
    __device__ float distance(const dfast382s& a, const dfast382s& b);
    inline
    __device__ float distance_s2(const dfast382s& a, const dfast382s& b);

    __device__ inline
    kernel_image2d<dfast382s>& previous_frame();
    __device__ inline
    kernel_image2d<dfast382s>& current_frame();
    __device__ inline
    kernel_image2d<i_float1>& pertinence();

  private:
    kernel_image2d<i_float1> pertinence_;
    kernel_image2d<dfast382s> f_prev_;
    kernel_image2d<dfast382s> f_;
  };

}

# include <cuimg/gpu/tracking/fast382s_feature.hpp>

#endif // ! CUIMG_FAST382S_FEATURE_H_
