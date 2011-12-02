#ifndef CUIMG_NAIVE_LOCAL_MATCHER_H_
# define  CUIMG_NAIVE_LOCAL_MATCHER_H_

# include <cuimg/gpu/image2d.h>

namespace cuimg
{

  template <typename F>
  class naive_local_matcher
  {
  public:
    typedef typename F::feature_t feature_t;
    typedef obox2d<point2d<int> > domain_t;

    struct particle
    {
      __host__ __device__
      particle() : age(0), speed(0.f, 0.f) {}
      __host__ __device__
      particle(int a, feature_t s, i_float2 speed) : age(a), state(s), speed(speed) {}
/*      __host__ __device__
      particle(const particle& o) : age(o.age), state(o.state), speed(o.speed) {}
      __host__ __device__
      particle& operator=(const particle& o) { age = o.age; state = o.state; speed = o.speed; return *this; }
*/
      int age;
      feature_t state;
      i_float2 speed;
    };

    naive_local_matcher(const domain_t& d);

    void update(F& f);

    const image2d<particle>& particles();

    void swap_buffers();

  private:

    image2d<particle> t1_;
    image2d<particle> t2_;

    image2d<particle>* particles_;
    image2d<particle>* new_particles_;

    image2d<i_float4> distance_;
    image2d<i_float4> test_;
    unsigned frame_cpt;
  };

}

# include <cuimg/gpu/tracking/naive_local_matcher.hpp>

#endif // !  CUIMG_NAIVE_LOCAL_MATCHER_H_
