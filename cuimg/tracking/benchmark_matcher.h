#ifndef CUIMG_BENCHMARK_MATCHER_H_
# define  CUIMG_BENCHMARK_MATCHER_H_


# include <vector>

# ifndef NO_CUDA
# include <thrust/host_vector.h>
# include <thrust/device_vector.h>
# endif

# include <cuimg/gpu/device_image2d.h>

# include <cuimg/tracking/global_mvt_thread.h>

# include <cuimg/target.h>
# include <cuimg/image2d_target.h>

namespace cuimg
{

  template <typename F, unsigned HS>
  class benchmark_matcher
  {
  public:
    enum { target = F::target };
    typedef typename F::feature_t feature_t;
    typedef obox2d domain_t;

    typedef image2d_target(target, i_short2) image2d_s2;
    typedef image2d_target(target, i_float4) image2d_f4;
    typedef image2d_target(target, char) image2d_c;
    typedef image2d_target(target, feature_t) image2d_F;


    struct particle
    {
      enum { history_size = HS };

      __host__ __device__
      particle() : age(0), fault(0), speed(0.f, 0.f) {}
      __host__ __device__
      particle(int a, feature_t s, i_float2 speed) : age(a), fault(0),
                                                     speed(speed) {}

      i_float2 speed;
      i_float2 brut_acceleration;

      i_short2 ipos;
      i_short2 pos;
      unsigned short age;
      unsigned short fault;
      float pertinence;
      i_short2 pos_history[history_size];
    };

    typedef typename thrust_vector<target, particle>::ret particle_vector;
    typedef typename thrust_vector<target, i_short2>::ret i_short2_vector;
    typedef image2d_target(target, particle) image2d_P;

    benchmark_matcher(const domain_t& d);

    void update(F& f, global_mvt_thread<benchmark_matcher<F, HS> >& t_mvt,
                const image2d_s2& ls_matches);

    void update(const flag<CPU>&, F& f, global_mvt_thread<benchmark_matcher<F, HS> >& t_mvt,
                const image2d_s2& ls_matches);

    void update(const flag<GPU>&, F& f, global_mvt_thread<benchmark_matcher<F, HS> >& t_mvt,
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
  };

}

# include <cuimg/tracking/benchmark_matcher.hpp>

#endif // !  CUIMG_BENCHMARK_MATCHER_H_
