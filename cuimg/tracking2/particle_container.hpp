#ifndef CUIMG_PARTICLE_CONTAINER_HP_
# define CUIMG_PARTICLE_CONTAINER_HP_

# include <cuimg/memset.h>
# include <cuimg/tracking2/compact_particles_image.h>

namespace cuimg
{

  particle::particle()
    : age(0), vpos(0), speed(0.f, 0.f)
  {
  }

  particle::particle(const particle& o)
      : speed(o.speed),
        vpos(o.vpos),
        age(o.age)
    {
    }

  // ################### Particle container methods ############
  // ###########################################################


  template <typename F,
	    template <class> class I>
  particle_container<F, I>::particle_container(const obox2d& d)
    : particles_img1_(d),
      particles_img2_(d),
      features1_(d),
      features2_(d),
      matches_(d)
  {
    particles_vec_.reserve((d.nrows() * d.ncols()) / 10);
    new_particles_img_ = &particles_img1_;
    cur_particles_img_ = &particles_img2_;
    new_features_ = &features1_;
    cur_features_ = &features2_;
  }

  template <typename F,
	    template <class> class I>
  const typename particle_container<F, I>::V&
  particle_container<F, I>::dense_particles()
  {
    return particles_vec_;
  }

  template <typename F,
	    template <class> class I>
  const I<particle>&
  particle_container<F, I>::sparse_particles()
  {
    return (*new_particles_img_);
  }

  template <typename F,
	    template <class> class I>
  const typename particle_container<F, I>::feature_type&
  particle_container<F, I>::feature_at(i_short2 p)
  {
    return (*cur_features_)(p);
  }

  template <typename F,
	    template <class> class I>
  const I<i_short2>&
  particle_container<F, I>::matches()
  {
    return matches_;
  }

  template <typename F,
	    template <class> class I>
  void
  particle_container<F, I>::before_matching()
  {
    // Reset matches, current particles and features.
    memset(matches_, 0);
    memset(*cur_particles_img_, 0);
    memset(*cur_features_, 0);
  }

  template <typename F,
	    template <class> class I>
  void
  particle_container<F, I>::after_matching()
  {
    // Swap new / current particles.
    std::swap(cur_features_, new_features_);
    std::swap(cur_particles_img_, new_particles_img_);
  }

  template <typename F,
	    template <class> class I>
  void
  particle_container<F, I>::after_new_particles()
  {
    compact_particles_image(*cur_particles_img_, particles_vec_);
  }

  template <typename F,
	    template <class> class I>
  void
  particle_container<F, I>::add(i_int2 p, const typename F::value_type& feature)
  {
    particle& pt = (*cur_particles_img_)(p);
    pt.age = 1;
    pt.speed = i_int2(0,0);

    (*cur_features_)(p) = feature;
  }

  template <typename F,
	    template <class> class I>
  void
  particle_container<F, I>::move(i_int2 src, i_int2 dst)
  {
    particle& p = (*new_particles_img_)(dst);
    p = (*cur_particles_img_)(src);

    p.age++;
    p.speed = dst - src;
    particles_vec_[p.vpos].set(p, dst);
    new_features_->operator()(dst) = cur_features_->operator()(src);

    matches_(src) = dst;
  }

  template <typename F,
	    template <class> class I>
  bool
  particle_container<F, I>::has(i_int2 p) const
  {
    return (*cur_particles_img_)(p).age > 0;
  }

}

#endif
