#ifndef CUIMG_PARTICLE_CONTAINER_HP_
# define CUIMG_PARTICLE_CONTAINER_HP_

# ifndef NO_CUDA
#  include <thrust/remove.h>
#  include <thrust/device_vector.h>
# endif

# include <cuimg/memset.h>
# include <cuimg/tracking2/compact_particles_image.h>
# include <cuimg/mt_apply.h>
# include <cuimg/builtin_math.h>
# include <cuimg/border.h>

namespace cuimg
{

  // ################### Particle container methods ############
  // ###########################################################

  template <typename F, typename P, typename A>
  particle_container<F, P, A>::particle_container(const obox2d& d)
    : sparse_buffer_(d, 7)
  {
    memset(sparse_buffer_, -1);
    fill_border_clamp(sparse_buffer_);
    frame_cpt_ = 0;
    particles_vec_.reserve((d.nrows() * d.ncols()) / 10);
    features_vec_.reserve((d.nrows() * d.ncols()) / 10);
  }

  template <typename F, typename P, typename A>
  particle_container<F, P, A>::particle_container(const particle_container<F, P, A>& d)
    : sparse_buffer_(d.domain())
  {
    copy(d.sparse_buffer_, sparse_buffer_);
    particles_vec_ = d.particles_vec_;
    features_vec_ = d.features_vec_;
    matches_ = d.matches_;
    compact_has_run_ = d.compact_has_run_;
    frame_cpt_ = d.frame_cpt_;
  }

  template <typename F, typename P, typename A>
  particle_container<F, P, A>&
  particle_container<F, P, A>::operator=(const particle_container& d)
  {
    if (not (sparse_buffer_.domain() == d.domain()))
      sparse_buffer_ = uint_image2d(d.domain());
    copy(d.sparse_buffer_, sparse_buffer_);
    particles_vec_ = d.particles_vec_;
    features_vec_ = d.features_vec_;
    matches_ = d.matches_;
    compact_has_run_ = d.compact_has_run_;
    frame_cpt_ = d.frame_cpt_;
    return *this;
  }

  template <typename F, typename P, typename A>
  typename particle_container<F, P, A>::particle_vector&
  particle_container<F, P, A>::dense_particles()
  {
    return particles_vec_;
  }

  template <typename F, typename P, typename A>
  typename particle_container<F, P, A>::uint_image2d&
  particle_container<F, P, A>::sparse_particles()
  {
    return sparse_buffer_;
  }


  template <typename F, typename P, typename A>
  void
  particle_container<F, P, A>::tick()
  {
    frame_cpt_++;
  }


  template <typename F, typename P, typename A>
  const typename particle_container<F, P, A>::particle_vector&
  particle_container<F, P, A>::dense_particles() const
  {
    return particles_vec_;
  }


  template <typename F, typename P, typename A>
  typename kernel_particle_container<F, P, A>::particle_type*
  kernel_particle_container<F, P, A>::dense_particles() const
  {
    return particles_vec_;
  }

  template <typename F, typename P, typename A>
  const typename particle_container<F, P, A>::uint_image2d&
  particle_container<F, P, A>::sparse_particles() const
  {
    return sparse_buffer_;
  }

  template <typename F, typename P, typename A>
  typename kernel_particle_container<F, P, A>::feature_type*
  kernel_particle_container<F, P, A>::features()
  {
    return features_vec_;
  }

  template <typename F, typename P, typename A>
  const typename particle_container<F, P, A>::feature_vector&
  particle_container<F, P, A>::features() const
  {
    return features_vec_;
  }

  template <typename F, typename P, typename A>
  const P&
  kernel_particle_container<F, P, A>::operator()(i_short2 p) const
  {
    assert(has(p));
    return particles_vec_[sparse_buffer_(p)];
  }


  template <typename F, typename P, typename A>
  const P&
  particle_container<F, P, A>::operator[](unsigned i) const
  {
    return particles_vec_[i];
  }

  template <typename F, typename P, typename A>
  P&
  particle_container<F, P, A>::operator[](unsigned i)
  {
    return particles_vec_[i];
  }

  // template <typename F, typename P, typename A>
  // const typename particle_container<F, P, A>::feature_type&
  // particle_container<F, P, A>::feature_at(i_short2 p)
  // {
  //   assert(has(p));
  //   return features_vec_[sparse_buffer_(p)];
  // }

  template <typename F, typename P, typename A>
  const typename particle_container<F, P, A>::uint_vector&
  particle_container<F, P, A>::matches() const
  {
    return matches_;
  }


#ifndef NO_CPP0X
  template <typename F, typename P, typename A>
  template <typename FC>
  void
  particle_container<F, P, A>::for_each_particle_st(FC func)
  {
    for (P& p : particles_vec_)
      if (p.age > 0) func(p);
  }

  template <typename F, typename P, typename A>
  template <typename FC>
  void
  particle_container<F, P, A>::for_each_particle_mt(FC func)
  {
#pragma omp parallel for schedule (static, 300)
    for (unsigned i = 0; i < particles_vec_.size(); i++)
      if (particles_vec_[i].age > 0) func(particles_vec_[i]);
  }
#endif

  template <typename F, typename P, typename A>
  void
  particle_container<F, P, A>::before_matching()
  {
    // Reset matches, new particles and features.
    memset(sparse_buffer_, -1);
    compact_has_run_ = false;
  }

  template <typename F, typename P, typename A>
  void
  particle_container<F, P, A>::after_matching()
  {
  }

  template <typename F, typename P, typename A>
  void
  particle_container<F, P, A>::after_new_particles()
  {
    compact();
  }

  template <typename F, typename P, typename A>
  void
  particle_container<F, P, A>::compact()
  {
    compact(A());
  }

  template <typename F, typename P, typename A>
  void
  particle_container<F, P, A>::compact(const cpu&)
  {
    SCOPE_PROF(compation);
    compact_has_run_ = true;

    matches_.resize(particles_vec_.size());
    std::fill(matches_.begin(), matches_.end(), -1);

    typename particle_vector::iterator pts_it = particles_vec_.begin();
    typename feature_vector::iterator feat_it = features_vec_.begin();
    typename particle_vector::iterator pts_res = particles_vec_.begin();
    typename feature_vector::iterator feat_res = features_vec_.begin();

    for (;pts_it != particles_vec_.end();)
    {
      if (pts_it->age != 0)
      {
        *pts_res++ = *pts_it;
        *feat_res++ = *feat_it;
        //int prev_index = sparse_buffer_(pts_it->pos);
        int prev_index = pts_it - particles_vec_.begin();
        int new_index = pts_res - particles_vec_.begin() - 1;
        sparse_buffer_(pts_it->pos) = new_index;
        matches_[prev_index] = new_index;
        assert(particles_vec_[sparse_buffer_(pts_it->pos)].pos == pts_it->pos);
      }

      pts_it++;
      feat_it++;
    }

    particles_vec_.resize(pts_res - particles_vec_.begin());
    features_vec_.resize(feat_res - features_vec_.begin());

    //compact_particles_image(*cur_particles_img_, particles_vec_);
  }

#ifdef NVCC

  template <typename T, typename P, typename F>
  __global__ void compact_update(T* uint_tmp_,
				 P* particles_vec_,
				 P* particles_vec_tmp_,
				 F* features_vec_,
				 F* features_vec_tmp_,
				 kernel_image2d<unsigned int> sparse_buffer_,
				 unsigned nparticles)
  {
    typedef unsigned int uint;
    unsigned int i = thread_pos1d();
    if (i >= nparticles) return;

    unsigned int old_index = uint_tmp_[i];
    assert(old_index != uint(-1));
    particles_vec_tmp_[i] = particles_vec_[old_index];
    features_vec_tmp_[i] = features_vec_[old_index];

    assert(sparse_buffer_.has(particles_vec_[old_index].pos));
    sparse_buffer_(particles_vec_[old_index].pos) = i;
  }

  template <typename F, typename P, typename A>
  void
  particle_container<F, P, A>::compact(const cuda_gpu&)
  {
    typedef unsigned int uint;
    using thrust::device_pointer_cast;
    // Thrust compaction.
    uint_tmp_.resize(sparse_buffer_.end() - sparse_buffer_.begin());
    thrust::detail::normal_iterator<thrust::device_ptr<unsigned int> >
      end =
      thrust::remove_copy(device_pointer_cast(sparse_buffer_.begin()), 
			  device_pointer_cast(sparse_buffer_.end()),
			  uint_tmp_.begin(), uint(-1));

    unsigned nparticles = end - uint_tmp_.begin();
    particles_vec_tmp_.resize(nparticles);
    features_vec_tmp_.resize(nparticles);

    if (nparticles)
      compact_update<<<A::dimgrid1d(nparticles), A::dimblock1d()>>>
	(thrust::raw_pointer_cast(uint_tmp_.data()),
	 thrust::raw_pointer_cast(particles_vec_.data()),
	 thrust::raw_pointer_cast(particles_vec_tmp_.data()),
	 thrust::raw_pointer_cast(features_vec_.data()),
	 thrust::raw_pointer_cast(features_vec_tmp_.data()),
	 sparse_buffer_,
	 nparticles);

    particles_vec_.swap(particles_vec_tmp_);
    features_vec_.swap(features_vec_tmp_);
  }

  template <typename P, typename F, typename FP>
  __global__ void init_new_particles(i_short2* new_points,
				     P* particles,
				     FP* features,
				     kernel_image2d<unsigned int> sparse_buffer_,
				     const F feature,
				     unsigned size,
				     unsigned old_size)
  {
    unsigned int i = thread_pos1d();
    if (i >= size) return;

    assert(sparse_buffer_.domain().has(new_points[i]));
    assert(new_points[i] != i_short2(0,0));
    P part;
    part.pos = new_points[i];
    part.age = 1;
    part.speed = i_int2(0,0);

    particles[i] = part;
    features[i] = feature(new_points[i]);
    sparse_buffer_(new_points[i]) = old_size + i;
  }

  template <typename F, typename P, typename A>
  void
  particle_container<F, P, A>::append_new_points(short2_image2d& new_points_, F& feature)
  {
    // Compact new_points.

    using thrust::device_pointer_cast;

    thrust::device_ptr<i_short2> end =
      thrust::remove(device_pointer_cast(new_points_.begin()),
		     device_pointer_cast(new_points_.end()), i_short2(0,0));

    unsigned old_size = particles_vec_.size();
    unsigned size = end - device_pointer_cast(new_points_.begin());

    assert(size < new_points_.domain().size());

    // Allocate space for new particles.
    particles_vec_.resize(particles_vec_.size() + size);
    features_vec_.resize(features_vec_.size() + size);

    // init new particles
    typedef const typename ::cuimg::kernel_type<F>::ret kernel_feature;
    if (size)
      init_new_particles<<<A::dimgrid1d(size), A::dimblock1d()>>>
	(new_points_.data(),
	 thrust::raw_pointer_cast(particles_vec_.data() + old_size),
	 thrust::raw_pointer_cast(features_vec_.data() + old_size),
	 mki(sparse_buffer_),
	 kernel_feature(feature), size, old_size);

  }

#endif

  template <typename F, typename P, typename A>
  void
  particle_container<F, P, A>::add(i_int2 p, const typename F::value_type& feature)
  {
    P pt;
    pt.age = 1;
    if (flow_(p) != NO_FLOW)
      pt.speed = flow_(p/8);
    else
    {
      pt.speed = i_int2(0,0);
    }
    // pt.speed = i_int2(0,0);
    pt.pos = p;
    pt.prev_match_time = frame_cpt_;
    pt.next_match_time = frame_cpt_ + 1;
    sparse_buffer_(p) = particles_vec_.size();
    particles_vec_.push_back(pt);
    features_vec_.push_back(feature);
  }

  template <typename F, typename P, typename A>
  void
  kernel_particle_container<F, P, A>::remove(int i)
  {
    P& p = particles_vec_[i];
    p.age = 0;
  }

  template <typename F, typename P, typename A>
  void
  particle_container<F, P, A>::remove(int i)
  {
    P& p = particles_vec_[i];
    p.age = 0;
  }

  template <typename F, typename P, typename A>
  void
  particle_container<F, P, A>::remove(const i_short2& pos)
  {
    P& p = particles_vec_[sparse_buffer_(p.pos)];
    p.age = 0;
  }

  template <typename F, typename P, typename A>
  void
  kernel_particle_container<F, P, A>::move(unsigned i, particle_coords dst, const feature_type& f)
  {
    assert(i < size_);
    assert(i >= 0);
    particle_type p = particles_vec_[i];
    assert(domain().has(p.pos));
    assert(domain().has(dst));
    p.age++;
    i_int2 new_speed = dst - p.pos;
    if (p.age > 1)
      p.acceleration = (new_speed - p.speed);
    else
      p.acceleration = i_int2(0,0);
    //p.speed = i_float2(new_speed) / (frame_cpt_ - p.prev_match_time);
    p.speed = i_float2(new_speed);
    p.pos = dst;
    // p.prev_match_time = frame_cpt_;
    // float period = 10.f;
    // i_float2 mesure = p.speed;
    // if (norml2(mesure) > 0.f)
    // {
    //   period = 10.f / norml2(mesure);
    //   period = period > 10.f ? 10.f : period;
    // }
    // if (period < 1)
    //   period = 1;
    // // p.next_match_time = frame_cpt_ + period;
    // p.next_match_time = frame_cpt_ + 1;

    particles_vec_[i] = p;
    if ((p.speed.y + p.speed.x) > 0)
      features_vec_[i] = f;

    sparse_buffer_(dst) = i;

    assert(domain().has(p.pos));
  }

  template <typename F, typename P, typename A>
  void
  kernel_particle_container<F, P, A>::touch(unsigned i)
  {
    assert(i < size_);
    assert(i >= 0);
    particle_type& p = particles_vec_[i];
    assert(domain().has(p.pos));

    p.age++;
    sparse_buffer_(p.pos) = i;
  }

  template <typename F, typename P, typename A>
  bool
  kernel_particle_container<F, P, A>::has(i_int2 p) const
  {
    unsigned int r = -1;
    return sparse_buffer_(p) != r;
  }


  template <typename F, typename P, typename A>
  void
  particle_container<F, P, A>::clear()
  {
    memset(sparse_buffer_, 255);
    particles_vec_.clear();
    features_vec_.clear();
    matches_.clear();
  }

  // template <typename T>
  // T* kernel_cast(T*, const cpu&)
  // {
  //   return
  // }

  template <typename F, typename P, typename A>
  kernel_particle_container<F, P, A>::kernel_particle_container(particle_container<F, P, A>& o)
    : sparse_buffer_(o.sparse_particles()),
#ifndef NO_CUDA
      particles_vec_(thrust::raw_pointer_cast(o.dense_particles().data())),
      features_vec_(thrust::raw_pointer_cast(o.features().data())),
#else
      particles_vec_(o.dense_particles().data()),
      features_vec_(o.features().data()),
#endif
      frame_cpt_(o.frame_cpt())
#ifndef NDEBUG
    , size_(o.dense_particles().size())
#endif
  {
  }

}

#endif
