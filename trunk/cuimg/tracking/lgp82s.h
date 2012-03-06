#ifndef CUIMG_LGP82S_FEATURE_H_
# define CUIMG_LGP82S_FEATURE_H_

# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{

  struct dlgp82s
  {
    __host__ __device__ dlgp82s() {}

    __host__ __device__ dlgp82s(const dlgp82s& o)
    {
      pertinence = o.pertinence;
      for (unsigned i = 0; i < 16; i++) distances[i] = o.distances[i];
    }

    __host__ __device__
    dlgp82s& operator=(const dlgp82s& o)
    {
      pertinence = o.pertinence;
      for (unsigned i = 0; i < 16; i++) distances[i] = o.distances[i];
      return *this;
    }

    float distances[16];
    float pertinence;
  };

  __host__ __device__ inline
  dlgp82s operator+(const dlgp82s& a, const dlgp82s& b);
  __host__ __device__ inline
  dlgp82s operator-(const dlgp82s& a, const dlgp82s& b);

  template <typename S>
  __host__ __device__ inline
  dlgp82s operator/(const dlgp82s& a, const S& s);

  template <typename S>
  __host__ __device__ inline
  dlgp82s operator*(const dlgp82s& a, const S& s);

  class kernel_lgp82s_feature;

  class lgp82s_feature
  {
  public:
    typedef dlgp82s feature_t;
    typedef obox2d<point2d<int> > domain_t;

    typedef kernel_lgp82s_feature kernel_type;

    inline lgp82s_feature(const domain_t& d);

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
    image2d_f1 blurred_s1_;
    image2d_f1 blurred_s2_;
    image2d_f1 tmp_;

    image2d_f1 pertinence_;

    image2d_D f1_;
    image2d_D f2_;

    image2d_D* f_prev_;
    image2d_D* f_;

    image2d_f4 lgp82s_color_;

    image2d_f4 color_blurred_;
    image2d_f4 color_tmp_;

    float grad_thresh;
  };

  class kernel_lgp82s_feature
  {
  public:
    typedef dlgp82s feature_t;

    inline kernel_lgp82s_feature(lgp82s_feature& f);


    inline
    __device__ float distance(const point2d<int>& p_prev,
                              const point2d<int>& p_cur);

    inline
    float distance(const dlgp82s& a, const dlgp82s& b);
    inline
    float distance_linear(const dlgp82s& a, const dlgp82s& b);

    __device__ inline
    kernel_image2d<dlgp82s>& previous_frame();
    __device__ inline
    kernel_image2d<dlgp82s>& current_frame();
    __device__ inline
    kernel_image2d<i_float1>& pertinence();

  private:
    kernel_image2d<i_float1> pertinence_;
    kernel_image2d<dlgp82s> f_prev_;
    kernel_image2d<dlgp82s> f_;
  };

}

# include <cuimg/tracking/lgp82s.hpp>

#endif // ! CUIMG_LGP82S_FEATURE_H_
