#ifndef CUIMG_PARTICLE_CONTAINER_H_
# define CUIMG_PARTICLE_CONTAINER_H_

# include <vector>
# include <cuimg/architectures.h>
# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  struct particle
  {
    typedef i_short2 coords_type;
    inline __host__ __device__
    particle() : age(0), speed(0.f, 0.f), fault(0) {}

  //   particle(const particle& o, i_short2 p)
  //     : speed(o.speed),
  //       pos(p),
	// age(o.age),
	// fault(0)
  //   {
  //   }

    // inline void set(const particle& o, i_short2 p)
    // {
    //   speed = o.speed;
    //   age = o.age;
    //   pos = p;
    //   fault = o.fault;
    // }

    i_float2 speed;
    i_short2 pos;
    i_short2 acceleration;
    unsigned short age;
    unsigned short fault;
    int prev_match_time;
    int next_match_time;
  };

  struct particle_f
  {
    typedef i_float2 coords_type;

    __host__ __device__
    particle_f() : age(0), speed(0.f, 0.f), fault(0) {}

    i_float2 speed;
    i_float2 pos;
    i_short2 acceleration;
    unsigned short age;
    unsigned short fault;
    int prev_match_time;
    int next_match_time;
  };

  template <typename F, typename P, typename A>
  class kernel_particle_container;

  template <typename F, typename P = particle, typename A = cpu>
  class particle_container
  {
  public:
    typedef typename P::coords_type particle_coords;
    typedef P particle_type;
    typedef typename A::template vector<particle_type>::ret particle_vector;
    typedef typename F::value_type feature_type;
    typedef typename A::template vector<feature_type>::ret feature_vector;
    typedef typename A::template image2d<unsigned int>::ret uint_image2d;
    typedef typename A::template image2d<i_short2>::ret short2_image2d;
    typedef typename A::template vector<unsigned int>::ret uint_vector;
    typedef typename A::template vector<i_short2>::ret short2_vector;
    typedef typename A::template image2d<i_float2>::ret flow_t;

    typedef A architecture;
    typedef kernel_particle_container<F, P, A> kernel_type;
    kernel_type to_kernel_type() { return *this; }

    particle_container(const obox2d& d);

    particle_container(const particle_container& d);
    particle_container& operator=(const particle_container& d);

    particle_vector& dense_particles();
    const particle_vector& dense_particles() const;
    uint_image2d& sparse_particles();
    const uint_image2d& sparse_particles() const;
    const feature_vector& features() const;
    feature_vector& features() { return features_vec_; }
    const uint_vector& matches() const;
    uint_vector& matches();

    //const feature_type& feature_at(i_short2 p);

    const particle_type& operator[](unsigned i) const;
    particle_type& operator[](unsigned i);

    template <typename FC>
    void for_each_particle_st(FC func);

    template <typename FC>
    void for_each_particle_mt(FC func);

    bool is_coherent();

    void compact();

    void set_flow(const flow_t& flow) { flow_ = flow; }
    const flow_t& flow() const { return flow_; }

#ifndef NO_CUDA
    void compact(const cuda_gpu&);
#endif

    void compact(const cpu&);

    void touch(unsigned i);
    void add(i_int2 p, const feature_type& f);
    void remove(int i);
    void remove(const i_short2& pos);
    void swap_buffers();
    inline const obox2d& domain() const { return sparse_buffer_.domain(); }
    void clear();
    void tick();
    unsigned frame_cpt() { return frame_cpt_; }

    void before_matching();
    void after_matching();
    void after_new_particles();

    void append_new_points(short2_image2d& new_points_, F& feature);

    inline bool compact_has_run() const { return compact_has_run_; }

    int size() const { return particles_vec_.size(); }

    template <typename T>
    struct default_die_fun
    {
      void operator()(const T&) {}
    };

    template <typename T>
    struct push_back_fun
    {
      push_back_fun(T& v) : t(&v) {}
      #ifndef NO_CPP0X
      void operator()(const typename T::value_type& v) { t->push_back(std::move(v)); }
      #else
      void operator()(const typename T::value_type& v) { t->push_back(v); }
      #endif

      T* t;
    };

#ifndef NO_CPP0X
    template <typename T, typename D = default_die_fun<typename T::value_type> >
    void sync_attributes(T& container, typename T::value_type new_value = typename T::value_type(),
			 D die_fun = default_die_fun<typename T::value_type>()) const;
#else
    template <typename T, typename D>
    void sync_attributes(T& container, typename T::value_type new_value = typename T::value_type(),
			 D die_fun = default_die_fun<typename T::value_type>()) const;
#endif

    template <typename T>
    void sync_attributes(T& container, T& dead_vectors, typename T::value_type new_value = typename T::value_type()) const;

  private:
    uint_image2d sparse_buffer_;
    flow_t flow_;
    particle_vector particles_vec_;
    particle_vector particles_vec_tmp_;
    short2_vector new_points_tmp_;
    uint_vector uint_tmp_;
    feature_vector features_vec_;
    feature_vector features_vec_tmp_;
    uint_vector matches_;
    bool compact_has_run_;
    unsigned frame_cpt_;
  };

  template <typename F, typename P = particle, typename A = cpu>
  class kernel_particle_container
  {
  public:
    typedef A architecture;
    typedef typename P::coords_type particle_coords;
    typedef P particle_type;
    typedef typename F::value_type feature_type;
    typedef typename A::template kernel_image2d<unsigned int>::ret uint_image2d;

    kernel_particle_container(particle_container<F, P, A>& c);

    inline __host__ __device__ void remove(int i);
    inline __host__ __device__ void remove(i_int2 p);
    inline __host__ __device__ particle_type* dense_particles() const;
    inline __host__ __device__ feature_type* features();
    inline __host__ __device__ void move(unsigned i, particle_coords dst, const feature_type& f);
    inline __host__ __device__ void touch(unsigned i);
    inline __host__ __device__ const particle_type& operator()(i_short2 p) const;
    inline __host__ __device__ bool has(i_int2 p) const;
    inline __host__ __device__ const obox2d& domain() const { return sparse_buffer_.domain(); }

    uint_image2d& sparse_particles() { return sparse_buffer_; }
    const uint_image2d& sparse_particles() const  { return sparse_buffer_; }

#ifndef NDEBUG
    inline __host__ __device__ int size() const { return size_; }
#endif

  private:
    uint_image2d sparse_buffer_;
    particle_type* particles_vec_;
    feature_type* features_vec_;
    unsigned int* matches_;
    unsigned frame_cpt_;
#ifndef NDEBUG
    unsigned size_;
#endif

  };

}

# include <cuimg/tracking2/particle_container.hpp>

#endif

