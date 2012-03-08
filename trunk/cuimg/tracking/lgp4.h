#ifndef CUIMG_LGP4_FEATURE_H_
# define CUIMG_LGP4_FEATURE_H_

# include <cuimg/gpu/device_image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{

  struct dlgp4
  {
    __host__ __device__ dlgp4() {}

    __host__ __device__ dlgp4(const dlgp4& o)
    {
      pertinence = o.pertinence;
      for (unsigned i = 0; i < 4; i++) distances[i] = o.distances[i];
    }

    __host__ __device__
    dlgp4& operator=(const dlgp4& o)
    {
      pertinence = o.pertinence;
      for (unsigned i = 0; i < 4; i++) distances[i] = o.distances[i];
      return *this;
    }

    float distances[4];
    float pertinence;
  };

  __host__ __device__ inline
  dlgp4 operator+(const dlgp4& a, const dlgp4& b);
  __host__ __device__ inline
  dlgp4 operator-(const dlgp4& a, const dlgp4& b);

  template <typename S>
  __host__ __device__ inline
  dlgp4 operator/(const dlgp4& a, const S& s);

  template <typename S>
  __host__ __device__ inline
  dlgp4 operator*(const dlgp4& a, const S& s);

  class kernel_lgp4_feature;

  class lgp4_feature
  {
  public:
    typedef dlgp4 feature_t;
    typedef obox2d<point2d<int> > domain_t;

    typedef kernel_lgp4_feature kernel_type;

    inline lgp4_feature(const domain_t& d);

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

    image2d_f4 lgp4_color_;

    image2d_f4 color_blurred_;
    image2d_f4 color_tmp_;

    float grad_thresh;
  };

  class kernel_lgp4_feature
  {
  public:
    typedef dlgp4 feature_t;

    inline kernel_lgp4_feature(lgp4_feature& f);


    inline
    __device__ float distance(const point2d<int>& p_prev,
                              const point2d<int>& p_cur);

    inline
    float distance(const dlgp4& a, const dlgp4& b);

    __device__ inline
    kernel_image2d<dlgp4>& previous_frame();
    __device__ inline
    kernel_image2d<dlgp4>& current_frame();
    __device__ inline
    kernel_image2d<i_float1>& pertinence();

  private:
    kernel_image2d<i_float1> pertinence_;
    kernel_image2d<dlgp4> f_prev_;
    kernel_image2d<dlgp4> f_;
  };

}

# include <cuimg/tracking/lgp4_feature.hpp>

#endif // ! CUIMG_LGP4_FEATURE_H_
