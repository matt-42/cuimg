#ifndef CUIMG_MULTI_SCALE_MATCHER_H_
# define  CUIMG_MULTI_SCALE_MATCHER_H_

# include <cuimg/gpu/image2d.h>

namespace cuimg
{

  template <typename F>
  class multi_scale_matcher
  {
  public:
    typedef typename F::feature_t feature_t;
    typedef obox2d<point2d<int> > domain_t;

    struct particle
    {
      __host__ __device__
      particle() : age(0), speed(0.f, 0.f) {}

      int age;
      feature_t state;
      i_float2 speed;
      i_float2 acceleration;
    };

    multi_scale_matcher(const domain_t& d);

    void update(F& f);

    const image2d<particle>& particles();
    const image2d_s2& matches();

    void swap_buffers();

  private:

    image2d<particle> t1_;
    image2d<particle> t2_;

    image2d<particle>* particles_;
    image2d<particle>* new_particles_;

    image2d_f4 distance_;
    image2d_f4 test_;
    image2d_f4 test2_;

    image2d_s2 matches_;
    unsigned frame_cpt;
  };

}

# include <cuimg/tracking/multi_scale_matcher.hpp>

#endif // !  CUIMG_MULTI_SCALE_MATCHER_H_
