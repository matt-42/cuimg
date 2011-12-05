#ifndef CUIMG_BFAST_FEATURE_H_
# define CUIMG_BFAST_FEATURE_H_

# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{

  struct dbfast
  {
    float max_diff;
    float intensity;
    char orientation;
    char sign;
    unsigned short tests;
    i_float4 color;
  };

  class kernel_bfast_feature;

  class bfast_feature
  {
  public:
    typedef dbfast feature_t;
    typedef obox2d<point2d<int> > domain_t;

    typedef kernel_bfast_feature kernel_type;

    inline bfast_feature(const domain_t& d);

    inline void update(const image2d<i_float4>& in);

    inline const domain_t& domain() const;

    inline image2d<dbfast>& previous_frame();
    inline image2d<dbfast>& current_frame();
    inline image2d<i_float1>& pertinence();

  private:
    inline void swap_buffers();


    image2d<i_float1> gl_frame_;
    image2d<i_float1> blurred_;
    image2d<i_float1> tmp_;

    image2d<i_float1> pertinence_;

    image2d<dbfast> f1_;
    image2d<dbfast> f2_;

    image2d<dbfast>* f_prev_;
    image2d<dbfast>* f_;

    image2d<i_float4> bfast_color_;

    float grad_thresh;
  };

  class kernel_bfast_feature
  {
  public:
    typedef dbfast feature_t;

    inline kernel_bfast_feature(bfast_feature& f);


    inline
    __device__ float distance(const point2d<int>& p_prev,
                              const point2d<int>& p_cur);

    inline
    float distance(const dbfast& a, const dbfast& b);

    __device__ inline
    kernel_image2d<dbfast>& previous_frame();
    __device__ inline
    kernel_image2d<dbfast>& current_frame();
    __device__ inline
    kernel_image2d<i_float1>& pertinence();

  private:
    kernel_image2d<i_float1> pertinence_;
    kernel_image2d<dbfast> f_prev_;
    kernel_image2d<dbfast> f_;
  };

}

# include <cuimg/gpu/tracking/bfast_feature.hpp>

#endif // ! CUIMG_BFAST_FEATURE_H_
