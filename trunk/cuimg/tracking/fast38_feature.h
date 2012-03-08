#ifndef CUIMG_FAST38_FEATURE_H_
# define CUIMG_FAST38_FEATURE_H_

# include <cuimg/gpu/device_image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{

  struct dfast38
  {
    __host__ __device__ dfast38() {}

    __host__ __device__ dfast38(const dfast38& o)
    {
      pertinence = o.pertinence;
      for (unsigned i = 0; i < 8; i++) distances[i] = o.distances[i];
    }

    __host__ __device__
    dfast38& operator=(const dfast38& o)
    {
      pertinence = o.pertinence;
      for (unsigned i = 0; i < 8; i++) distances[i] = o.distances[i];
      return *this;
    }

    float distances[8];
    float pertinence;
  };

  __host__ __device__ inline
  dfast38 operator+(const dfast38& a, const dfast38& b);
  __host__ __device__ inline
  dfast38 operator-(const dfast38& a, const dfast38& b);

  template <typename S>
  __host__ __device__ inline
  dfast38 operator/(const dfast38& a, const S& s);

  template <typename S>
  __host__ __device__ inline
  dfast38 operator*(const dfast38& a, const S& s);

  class kernel_fast38_feature;

  class fast38_feature
  {
  public:
    typedef dfast38 feature_t;
    typedef obox2d<point2d<int> > domain_t;

    typedef kernel_fast38_feature kernel_type;

    inline fast38_feature(const domain_t& d);

    inline void update(const image2d_f4& in);
    inline void update(const image2d_f1& in);

    void display() const;

    inline const domain_t& domain() const;

    inline image2d_D& previous_frame();
    inline image2d_D& current_frame();
    inline image2d_f1& pertinence();

    const image2d_f4& feature_color() const;


  private:
    inline void swap_buffers();


    image2d_f1 gl_frame_;
    image2d_f1 blurred_;
    image2d_f1 tmp_;

    image2d_f1 pertinence_;

    image2d_D f1_;
    image2d_D f2_;

    image2d_D* f_prev_;
    image2d_D* f_;

    image2d_f4 fast38_color_;

    image2d_f4 color_blurred_;
    image2d_f4 color_tmp_;

    float grad_thresh;
  };

  class kernel_fast38_feature
  {
  public:
    typedef dfast38 feature_t;

    inline kernel_fast38_feature(fast38_feature& f);


    inline
    __device__ float distance(const point2d<int>& p_prev,
                              const point2d<int>& p_cur);

    inline
    float distance(const dfast38& a, const dfast38& b);

    inline
    float distance_linear(const dfast38& a, const dfast38& b);

    __device__ inline
    kernel_image2d<dfast38>& previous_frame();
    __device__ inline
    kernel_image2d<dfast38>& current_frame();
    __device__ inline
    kernel_image2d<i_float1>& pertinence();

  private:
    kernel_image2d<i_float1> pertinence_;
    kernel_image2d<dfast38> f_prev_;
    kernel_image2d<dfast38> f_;
  };

}

# include <cuimg/tracking/fast38_feature.hpp>

#endif // ! CUIMG_FAST38_FEATURE_H_
