#ifndef CUIMG_PARTICLE_CONTAINER_H_
# define CUIMG_PARTICLE_CONTAINER_H_

# include <vector>
# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  struct particle
  {
    __host__ __device__ particle();

    particle(const particle& o);

    i_float2 speed;
    unsigned short vpos;
    unsigned short age;
  };

  struct particle_p
  {
    __host__ __device__
    particle_p() : age(0), speed(0.f, 0.f) {}

    particle_p(const particle& o, i_short2 p)
      : speed(o.speed),
        pos(p),
        age(o.age)
    {
    }

    inline void set(const particle& o, i_short2 p)
    {
      speed = o.speed;
      age = o.age;
      pos = p;
    }

    i_float2 speed;
    i_short2 pos;
    unsigned short age;
  };

  template <typename F,
	    template <class> class I = host_image2d>
  class particle_container
  {
  public:
    typedef std::vector<particle_p> V;
    typedef typename F::value_type feature_type;
    particle_container(const obox2d& d);

    const V& dense_particles();
    const I<particle>& sparse_particles();
    const I<feature_type>& features();
    const I<i_short2>& matches();

    const feature_type& feature_at(i_short2 p);

    void compact();
    void move(i_int2 src, i_int2 dst);
    bool has(i_int2 p) const;
    void add(i_int2 p, const feature_type& f);
    void swap_buffers();

    void before_matching();
    void after_matching();
    void after_new_particles();

  private:
    I<particle> particles_img1_;
    I<particle> particles_img2_;
    I<particle>* new_particles_img_, *cur_particles_img_;
    V particles_vec_;
    I<feature_type> features1_;
    I<feature_type> features2_;
    I<feature_type>* new_features_, *cur_features_;
    I<i_short2> matches_;
  };

}

# include <cuimg/tracking2/particle_container.hpp>

#endif

