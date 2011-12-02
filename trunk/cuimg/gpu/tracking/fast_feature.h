#ifndef CUIMG_FAST_FEATURE_H_
# define CUIMG_FAST_FEATURE_H_

# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{

  struct __align__(8) dfast
  {
    float max_diff;
    float intensity;
    char orientation;
    char sign;
  };

  class kernel_fast_feature;

  class fast_feature
  {
  public:
    typedef dfast feature_t;
    typedef obox2d<point2d<int> > domain_t;

    typedef kernel_fast_feature kernel_type;

    inline fast_feature(const domain_t& d);

    inline void update(const image2d<i_float4>& in);

    inline const domain_t& domain() const;

    inline image2d<dfast>& previous_frame();
    inline image2d<dfast>& current_frame();
    inline image2d<i_float1>& pertinence();

  private:
    inline void swap_buffers();


    image2d<i_float1> gl_frame_;
    image2d<i_float1> blurred_;
    image2d<i_float1> tmp_;

    image2d<i_float1> pertinence_;

    image2d<dfast> f1_;
    image2d<dfast> f2_;

    image2d<dfast>* f_prev_;
    image2d<dfast>* f_;

    image2d<i_float4> fast_color_;

    float grad_thresh;
  };

  class kernel_fast_feature
  {
  public:
    typedef dfast feature_t;

    inline kernel_fast_feature(fast_feature& f);


    inline
    __device__ float distance(const point2d<int>& p_prev,
                              const point2d<int>& p_cur);

    inline
    float distance(const dfast& a, const dfast& b);

    __device__ inline
    kernel_image2d<dfast>& previous_frame();
    __device__ inline
    kernel_image2d<dfast>& current_frame();
    __device__ inline
    kernel_image2d<i_float1>& pertinence();

  private:
    kernel_image2d<i_float1> pertinence_;
    kernel_image2d<dfast> f_prev_;
    kernel_image2d<dfast> f_;
  };

}

# include <cuimg/gpu/tracking/fast_feature.hpp>

#endif // ! CUIMG_FAST_FEATURE_H_
