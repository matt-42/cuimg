#ifndef CUIMG_NAIVE_LOCAL_MATCHER2_H_
# define  CUIMG_NAIVE_LOCAL_MATCHER2_H_

# include <cuimg/gpu/image2d.h>

namespace cuimg
{

  template <typename F>
  class naive_local_matcher2
  {
  public:
    typedef typename F::feature_t feature_t;
    typedef obox2d<point2d<int> > domain_t;

    struct particle
    {
      __host__ __device__
      particle() : age(0), fault(0), speed(0.f, 0.f) {}
      __host__ __device__
      particle(int a, feature_t s, i_float2 speed) : age(a), fault(0), state(s), speed(speed) {}

      feature_t state;

      i_float2 speed;
      //i_float2 acceleration;
      i_float2 brut_acceleration;

      i_short2 ipos;
      unsigned short age;
      unsigned short fault;
      i_int2 __padding__;
    };

    naive_local_matcher2(const domain_t& d);

    void update(F& f, i_short2 mvt, const image2d<i_short2>& ls_matches);
    void display() const;

    const image2d<particle>& particles();
    const image2d<i_short2>& matches();
    const image2d<char>& errors() const;

    void swap_buffers();

  private:

    image2d<particle> t1_;
    image2d<particle> t2_;

    image2d<particle>* particles_;
    image2d<particle>* new_particles_;

    image2d<i_float4> distance_;
    image2d<i_float4> test_;
    image2d<i_float4> test2_;
    image2d<char> errors_;
    image2d<i_float4> fm_disp_;

    image2d<i_short2> matches_;
    unsigned frame_cpt;
  };

}

# include <cuimg/gpu/tracking/naive_local_matcher2.hpp>

#endif // !  CUIMG_NAIVE_LOCAL_MATCHER2_H_
