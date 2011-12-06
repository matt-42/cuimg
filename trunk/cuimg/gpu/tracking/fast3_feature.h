#ifndef CUIMG_FAST3_FEATURE_H_
# define CUIMG_FAST3_FEATURE_H_

# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{

  struct dfast3
  {
    i_float4 distances;
    float pertinence;
  };

  __host__ __device__ inline
  inline dfast3 operator+(const dfast3& a, const dfast3& b);
  __host__ __device__ inline
  inline dfast3 operator-(const dfast3& a, const dfast3& b);

  template <typename S>
  __host__ __device__ inline
  dfast3 operator/(const dfast3& a, const S& s);

  template <typename S>
  __host__ __device__ inline
  dfast3 operator*(const dfast3& a, const S& s);

  class kernel_fast3_feature;

  class fast3_feature
  {
  public:
    typedef dfast3 feature_t;
    typedef obox2d<point2d<int> > domain_t;

    typedef kernel_fast3_feature kernel_type;

    inline fast3_feature(const domain_t& d);

    inline void update(const image2d<i_float4>& in);

    inline const domain_t& domain() const;

    inline image2d<dfast3>& previous_frame();
    inline image2d<dfast3>& current_frame();
    inline image2d<i_float1>& pertinence();

    const image2d<i_float4>& feature_color() const;


  private:
    inline void swap_buffers();


    image2d<i_float1> gl_frame_;
    image2d<i_float1> blurred_;
    image2d<i_float1> tmp_;

    image2d<i_float1> pertinence_;

    image2d<dfast3> f1_;
    image2d<dfast3> f2_;

    image2d<dfast3>* f_prev_;
    image2d<dfast3>* f_;

    image2d<i_float4> fast3_color_;

    float grad_thresh;
  };

  class kernel_fast3_feature
  {
  public:
    typedef dfast3 feature_t;

    inline kernel_fast3_feature(fast3_feature& f);


    inline
    __device__ float distance(const point2d<int>& p_prev,
                              const point2d<int>& p_cur);

    inline
    float distance(const dfast3& a, const dfast3& b);

    __device__ inline
    kernel_image2d<dfast3>& previous_frame();
    __device__ inline
    kernel_image2d<dfast3>& current_frame();
    __device__ inline
    kernel_image2d<i_float1>& pertinence();

  private:
    kernel_image2d<i_float1> pertinence_;
    kernel_image2d<dfast3> f_prev_;
    kernel_image2d<dfast3> f_;
  };

}

# include <cuimg/gpu/tracking/fast3_feature.hpp>

#endif // ! CUIMG_FAST3_FEATURE_H_
