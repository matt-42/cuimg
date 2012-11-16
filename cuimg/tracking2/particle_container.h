#ifndef CUIMG_PARTICLE_CONTAINER_H_
# define CUIMG_PARTICLE_CONTAINER_H_

# include <vector>
# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  struct particle
  {
    __host__ __device__
    particle() : age(0), speed(0.f, 0.f) {}

    particle(const particle& o, i_short2 p)
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
    typedef std::vector<particle> V;
    typedef typename F::value_type feature_type;
    typedef std::vector<feature_type> FV;
    particle_container(const obox2d& d);

    V& dense_particles();
    I<unsigned short>& sparse_particles();
    const FV& features();
    const I<i_short2>& matches();

    const feature_type& feature_at(i_short2 p);

    const particle& operator()(i_short2 p) const;
    particle& operator()(i_short2 p);
    const particle& operator[](unsigned i) const;
    particle& operator[](unsigned i);

    bool is_coherent();

    void compact();
    void move(unsigned i, i_int2 dst);
    bool has(i_int2 p) const;
    void add(i_int2 p, const feature_type& f);
    void remove(const i_short2& pos);
    void remove(int i);
    void swap_buffers();

    void before_matching();
    void after_matching();
    void after_new_particles();

  private:
    I<unsigned short> sparse_buffer_;
    V particles_vec_;
    FV features_vec_;
    I<i_short2> matches_;
  };

}

# include <cuimg/tracking2/particle_container.hpp>

#endif

