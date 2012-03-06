#ifndef CUIMG_LGP8_FEATURE_H_
# define CUIMG_LGP8_FEATURE_H_

# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{

  struct dlgp8
  {
    __host__ __device__ dlgp8() {}

    __host__ __device__ dlgp8(const dlgp8& o)
    {
      pertinence = o.pertinence;
      for (unsigned i = 0; i < 8; i++) distances[i] = o.distances[i];
    }

    __host__ __device__
    dlgp8& operator=(const dlgp8& o)
    {
      pertinence = o.pertinence;
      for (unsigned i = 0; i < 8; i++) distances[i] = o.distances[i];
      return *this;
    }

    float distances[8];
    float pertinence;
  };

  __host__ __device__ inline
  dlgp8 operator+(const dlgp8& a, const dlgp8& b);
  __host__ __device__ inline
  dlgp8 operator-(const dlgp8& a, const dlgp8& b);

  template <typename S>
  __host__ __device__ inline
  dlgp8 operator/(const dlgp8& a, const S& s);

  template <typename S>
  __host__ __device__ inline
  dlgp8 operator*(const dlgp8& a, const S& s);

  class kernel_lgp8_feature;

  class lgp8_feature
  {
  public:
    typedef dlgp8 feature_t;
    typedef obox2d<point2d<int> > domain_t;

    typedef kernel_lgp8_feature kernel_type;

    inline lgp8_feature(const domain_t& d);

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

    image2d_f4 lgp8_color_;

    image2d_f4 color_blurred_;
    image2d_f4 color_tmp_;

    float grad_thresh;
  };

  class kernel_lgp8_feature
  {
  public:
    typedef dlgp8 feature_t;

    inline kernel_lgp8_feature(lgp8_feature& f);


    inline
    __device__ float distance(const point2d<int>& p_prev,
                              const point2d<int>& p_cur);

    inline
    float distance(const dlgp8& a, const dlgp8& b);
    inline
    float distance_linear(const dlgp8& a, const dlgp8& b);

    __device__ inline
    kernel_image2d<dlgp8>& previous_frame();
    __device__ inline
    kernel_image2d<dlgp8>& current_frame();
    __device__ inline
    kernel_image2d<i_float1>& pertinence();

  private:
    kernel_image2d<i_float1> pertinence_;
    kernel_image2d<dlgp8> f_prev_;
    kernel_image2d<dlgp8> f_;
  };

}

# include <cuimg/tracking/lgp8.hpp>

#endif // ! CUIMG_LGP8_FEATURE_H_
