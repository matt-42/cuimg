#ifndef CUIMG_FAST16_2S_FEATURE_H_
# define CUIMG_FAST16_2S_FEATURE_H_

# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{

  struct dfast16_2s
  {
    __host__ __device__ dfast16_2s() {}

    __host__ __device__ dfast16_2s(const dfast16_2s& o)
    {
      pertinence = o.pertinence;
      for (unsigned i = 0; i < 32; i++) distances[i] = o.distances[i];
    }

    __host__ __device__
    dfast16_2s& operator=(const dfast16_2s& o)
    {
      pertinence = o.pertinence;
      for (unsigned i = 0; i < 32; i++) distances[i] = o.distances[i];
      return *this;
    }

    float distances[32];
    float pertinence;
    //char distances[16];
  };

  __host__ __device__ inline
  dfast16_2s operator+(const dfast16_2s& a, const dfast16_2s& b);
  __host__ __device__ inline
  dfast16_2s operator-(const dfast16_2s& a, const dfast16_2s& b);

  template <typename S>
  __host__ __device__ inline
  dfast16_2s operator/(const dfast16_2s& a, const S& s);

  template <typename S>
  __host__ __device__ inline
  dfast16_2s operator*(const dfast16_2s& a, const S& s);

  class kernel_fast16_2s_feature;

  class fast16_2s_feature
  {
  public:
    typedef dfast16_2s feature_t;
    typedef obox2d<point2d<int> > domain_t;

    typedef kernel_fast16_2s_feature kernel_type;

    inline fast16_2s_feature(const domain_t& d);

    inline void update(const image2d_f4& in);
    inline void update(const image2d_f1& in);

    inline const domain_t& domain() const;

    inline image2d_D& previous_frame();
    inline image2d_D& current_frame();
    inline image2d_f1& pertinence();

    const image2d_f4& feature_color() const;

    void display() const;

  private:
    inline void swap_buffers();


    image2d_f1 gl_frame_;
    image2d_f1 blurred_s1_;
    image2d_f1 blurred_s2_;
    image2d_f1 tmp_;

    image2d_f1 pertinence_;

    image2d_D f1_;
    image2d_D f2_;

    image2d_D* f_prev_;
    image2d_D* f_;

    image2d_f4 fast162s_color_;

    image2d_f4 color_blurred_;
    image2d_f4 color_tmp_;

    float grad_thresh;
  };

  class kernel_fast16_2s_feature
  {
  public:
    typedef dfast16_2s feature_t;

    inline kernel_fast16_2s_feature(fast16_2s_feature& f);


    inline
    __device__ float distance(const point2d<int>& p_prev,
                              const point2d<int>& p_cur);

    inline
    __device__ float distance_linear(const dfast16_2s& a, const dfast16_2s& b);
    inline
    __device__ float distance(const dfast16_2s& a, const dfast16_2s& b);
    inline
    __device__ float distance_s2(const dfast16_2s& a, const dfast16_2s& b);

    __device__ inline
    kernel_image2d<dfast16_2s>& previous_frame();
    __device__ inline
    kernel_image2d<dfast16_2s>& current_frame();
    __device__ inline
    kernel_image2d<i_float1>& pertinence();

  private:
    kernel_image2d<i_float1> pertinence_;
    kernel_image2d<dfast16_2s> f_prev_;
    kernel_image2d<dfast16_2s> f_;
  };

}

# include <cuimg/tracking/fast16_2s_feature.hpp>

#endif // ! CUIMG_FAST16_2S_FEATURE_H_
