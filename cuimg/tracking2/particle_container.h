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
    particle() : age(0), speed(0.f, 0.f), fault(0) {}

    particle(const particle& o, i_short2 p)
      : speed(o.speed),
        pos(p),
	age(o.age),
	fault(0)
    {
    }

    inline void set(const particle& o, i_short2 p)
    {
      speed = o.speed;
      age = o.age;
      pos = p;
      fault = o.fault;
    }

    i_float2 speed;
    i_short2 pos;
    i_short2 acceleration;
    unsigned short age;
    unsigned short fault;
  };

  template <typename F, typename P = particle,
	    template <class> class I = host_image2d>
  class particle_container
  {
  public:
    typedef P particle_type;
    typedef std::vector<particle_type> V;
    typedef typename F::value_type feature_type;
    typedef std::vector<feature_type> FV;
    particle_container(const obox2d& d);

    V& dense_particles();
    const V& dense_particles() const;
    I<unsigned short>& sparse_particles();
    const I<unsigned short>& sparse_particles() const;
    const FV& features();
    const std::vector<unsigned int>& matches() const;

    const feature_type& feature_at(i_short2 p);

    const particle_type& operator()(i_short2 p) const;
    particle_type& operator()(i_short2 p);
    const particle_type& operator[](unsigned i) const;
    particle_type& operator[](unsigned i);

    template <typename FC>
    void for_each_particle_st(FC func);

    template <typename FC>
    void for_each_particle_mt(FC func);

    bool is_coherent();

    void compact();
    void move(unsigned i, i_int2 dst, const feature_type& f);
    bool has(i_int2 p) const;
    void add(i_int2 p, const feature_type& f);
    void remove(const i_short2& pos);
    void remove(int i);
    void swap_buffers();
    inline const obox2d& domain() { return sparse_buffer_.domain(); }
    void clear();

    void before_matching();
    void after_matching();
    void after_new_particles();

    inline bool compact_has_run() const { return compact_has_run_; }
  private:
    I<unsigned int> sparse_buffer_;
    V particles_vec_;
    FV features_vec_;
    std::vector<unsigned int> matches_;
    bool compact_has_run_;
  };

}

# include <cuimg/tracking2/particle_container.hpp>

#endif

