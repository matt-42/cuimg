#ifndef CUIMG_NAIVE_LOCAL_MATCHER_THRUST_H_
# define  CUIMG_NAIVE_LOCAL_MATCHER_THRUST_H_

# include <thrust/device_vector.h>

# include <cuimg/gpu/image2d.h>

# include <cuimg/tracking/global_mvt_thread.h>

namespace cuimg
{

  template <typename F>
  class naive_local_matcher_thrust
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

    naive_local_matcher_thrust(const domain_t& d);

    void update(F& f, global_mvt_thread<naive_local_matcher_thrust<F> >& t_mvt,
                const image2d_s2& ls_matches);
    void display() const;

    const image2d<particle>& particles() const;
    const image2d_s2& matches() const;
    const image2d_c& errors() const;

    void swap_buffers();

    const thrust::device_vector<particle>& compact_particles() const;

    unsigned n_particles() const;

  private:

    image2d<particle> t1_;
    image2d<particle> t2_;

    image2d<particle>* particles_;
    image2d<particle>* new_particles_;

    image2d_f4 distance_;
    image2d_f4 test_;
    image2d_f4 test2_;
    image2d_c errors_;
    image2d_f4 fm_disp_;

    image2d_s2 matches_;

    thrust::device_vector<i_short2> particles_vec1_;
    thrust::device_vector<i_short2> particles_vec2_;
    thrust::device_vector<particle> compact_particles_;

    unsigned n_particles_;
    unsigned frame_cpt;
  };

}

# include <cuimg/tracking/naive_local_matcher_thrust.hpp>

#endif // !  CUIMG_NAIVE_LOCAL_MATCHER_THRUST_H_
