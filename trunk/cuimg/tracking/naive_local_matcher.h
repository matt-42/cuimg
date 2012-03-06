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

      int age;
      feature_t state;
      i_float2 speed;
      i_float2 acceleration;
    };

    naive_local_matcher(const domain_t& d);

    void update(F& f, i_short2 mvt, const image2d_s2& ls_matches);
    void display() const;

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

# include <cuimg/tracking/naive_local_matcher.hpp>

#endif // !  CUIMG_NAIVE_LOCAL_MATCHER_H_
