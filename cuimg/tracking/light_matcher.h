#ifndef CUIMG_LIGHT_MATCHER_H_
# define  CUIMG_LIGHT_MATCHER_H_


# include <vector>

# ifndef NO_CUDA
# include <thrust/host_vector.h>
# include <thrust/device_vector.h>
# endif

# include <cuimg/gpu/device_image2d.h>

# include <cuimg/tracking/global_mvt_thread.h>

# include <cuimg/target.h>
# include <cuimg/image2d_target.h>
# include <cuimg/tracking/matcher_tools.h>

namespace cuimg
{

  template <typename F>
  class light_matcher
  {
  public:
    static const cuimg::target target = F::target;
    //enum { target = F::target };
    typedef typename F::feature_t feature_t;
    typedef obox2d domain_t;

    typedef image2d_target(target, i_short2) image2d_s2;
    typedef image2d_target(target, i_float4) image2d_f4;
    typedef image2d_target(target, char) image2d_c;
    typedef image2d_target(target, feature_t) image2d_F;


    struct particle
    {
      __host__ __device__
      particle() : age(0), fault(0), speed(0.f, 0.f) {}
      __host__ __device__
      particle(int a, feature_t s, i_float2 speed) : speed(speed), age(a), fault(0)
      {}

      particle(const particle& o)
      : speed(o.speed),
        brut_acceleration(o.brut_acceleration),
        pos(o.pos),
        age(o.age),
        fault(o.fault)
      {
      }

      i_float2 speed;
      i_float2 brut_acceleration;
      i_short2 pos;
      unsigned short age;
      unsigned short fault;
    };

    typedef typename thrust_vector<target, particle>::ret particle_vector;
    typedef typename thrust_vector<target, i_short2>::ret i_short2_vector;
    typedef image2d_target(target, particle) image2d_P;

    light_matcher(const domain_t& d);
    ~light_matcher();

    void update(F& f, global_mvt_thread<light_matcher<F> >& t_mvt,
                const image2d_s2& ls_matches);

    void update(const flag<CPU>&, F& f, global_mvt_thread<light_matcher<F> >& t_mvt,
                const image2d_s2& ls_matches);

    void update(const flag<GPU>&, F& f, global_mvt_thread<light_matcher<F> >& t_mvt,
                const image2d_s2& ls_matches);

    void display() const;

    const image2d_P& particles() const;
    const image2d_s2& matches() const;
    const image2d_c& errors() const;

    void swap_buffers();

    particle_vector& compact_particles();
    const particle_vector& compact_particles() const;
    const i_short2_vector& particle_positions() const { return particles_vec1_; }

    unsigned n_particles() const;

  private:
    void particles_compation(const flag<CPU>&);
    void particles_compation_old(const flag<CPU>&);

    image2d_P t1_;
    image2d_P t2_;

    image2d_P* particles_;
    image2d_P* new_particles_;

    image2d_f4 distance_;
    image2d_f4 test_;
    image2d_f4 test2_;
    image2d_c errors_;
    image2d_f4 fm_disp_;

    image2d_s2 matches_;

    image2d_F states_;

    i_short2_vector particles_vec1_;
    i_short2_vector particles_vec2_;
    particle_vector compact_particles_;

    int n_particles_;
    unsigned frame_cpt;

    i_short2** compaction_openmp_buffers;
  };

}

# include <cuimg/tracking/light_matcher.hpp>

#endif // !  CUIMG_LIGHT_MATCHER_H_
