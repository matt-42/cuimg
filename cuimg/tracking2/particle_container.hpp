#ifndef CUIMG_PARTICLE_CONTAINER_HP_
# define CUIMG_PARTICLE_CONTAINER_HP_

# include <cuimg/memset.h>
# include <cuimg/tracking2/compact_particles_image.h>
# include <cuimg/mt_apply.h>
# include <cuimg/builtin_math.h>

namespace cuimg
{

  // ################### Particle container methods ############
  // ###########################################################

  template <typename F, typename P,
	    template <class> class I>
  particle_container<F, P, I>::particle_container(const obox2d& d)
    : sparse_buffer_(d)
  {
    frame_cpt = 0;
    particles_vec_.reserve((d.nrows() * d.ncols()) / 10);
    features_vec_.reserve((d.nrows() * d.ncols()) / 10);
  }

  template <typename F, typename P,
	    template <class> class I>
  particle_container<F, P, I>::particle_container(const particle_container<F, P, I>& d)
    : sparse_buffer_(d.domain())
  {
    copy(d.sparse_buffer_, sparse_buffer_);
    particles_vec_ = d.particles_vec_;
    features_vec_ = d.features_vec_;
    matches_ = d.matches_;
    compact_has_run_ = d.compact_has_run_;
    frame_cpt = d.frame_cpt;
  }

  template <typename F, typename P,
	    template <class> class I>
  particle_container<F, P, I>&
  particle_container<F, P, I>::operator=(const particle_container& d)
  {
    if (not (sparse_buffer_.domain() == d.domain()))
      sparse_buffer_ = I<unsigned int>(d.domain());
    copy(d.sparse_buffer_, sparse_buffer_);
    particles_vec_ = d.particles_vec_;
    features_vec_ = d.features_vec_;
    matches_ = d.matches_;
    compact_has_run_ = d.compact_has_run_;
    frame_cpt = d.frame_cpt;
    return *this;
  }

  template <typename F, typename P,
	    template <class> class I>
  typename particle_container<F, P, I>::V&
  particle_container<F, P, I>::dense_particles()
  {
    return particles_vec_;
  }

  template <typename F, typename P,
	    template <class> class I>
  I<unsigned int>&
  particle_container<F, P, I>::sparse_particles()
  {
    return sparse_buffer_;
  }


  template <typename F, typename P,
	    template <class> class I>
  void
  particle_container<F, P, I>::tick()
  {
    frame_cpt++;
  }


  template <typename F, typename P,
	    template <class> class I>
  const typename particle_container<F, P, I>::V&
  particle_container<F, P, I>::dense_particles() const
  {
    return particles_vec_;
  }

  template <typename F, typename P,
	    template <class> class I>
  const I<unsigned int>&
  particle_container<F, P, I>::sparse_particles() const
  {
    return sparse_buffer_;
  }

  template <typename F, typename P,
	    template <class> class I>
  const typename particle_container<F, P, I>::FV&
  particle_container<F, P, I>::features()
  {
    return features_vec_;
  }

  template <typename F, typename P,
	    template <class> class I>
  const P&
  particle_container<F, P, I>::operator()(i_short2 p) const
  {
    assert(has(p));
    return particles_vec_[sparse_buffer_(p)];
  }

  template <typename F, typename P,
	    template <class> class I>
  P&
  particle_container<F, P, I>::operator()(i_short2 p)
  {
    assert(has(p));
    return particles_vec_[sparse_buffer_(p)];
  }

  template <typename F, typename P,
	    template <class> class I>
  const P&
  particle_container<F, P, I>::operator[](unsigned i) const
  {
    return particles_vec_[i];
  }

  template <typename F, typename P,
	    template <class> class I>
  P&
  particle_container<F, P, I>::operator[](unsigned i)
  {
    return particles_vec_[i];
  }

  template <typename F, typename P,
	    template <class> class I>
  const typename particle_container<F, P, I>::feature_type&
  particle_container<F, P, I>::feature_at(i_short2 p)
  {
    assert(has(p));
    return features_vec_[sparse_buffer_(p)];
  }

  template <typename F, typename P,
	    template <class> class I>
  const std::vector<unsigned int>&
  particle_container<F, P, I>::matches() const
  {
    return matches_;
  }


  template <typename F, typename P,
	    template <class> class I>
  template <typename FC>
  void
  particle_container<F, P, I>::for_each_particle_st(FC func)
  {
    for (P& p : particles_vec_)
      if (p.age > 0) func(p);
  }

  template <typename F, typename P,
	    template <class> class I>
  template <typename FC>
  void
  particle_container<F, P, I>::for_each_particle_mt(FC func)
  {
#pragma omp parallel for schedule (static, 300)
    for (unsigned i = 0; i < particles_vec_.size(); i++)
      if (particles_vec_[i].age > 0) func(particles_vec_[i]);
  }

  template <typename F, typename P,
	    template <class> class I>
  void
  particle_container<F, P, I>::before_matching()
  {
    // Reset matches, new particles and features.
    memset(sparse_buffer_, 255);
    compact_has_run_ = false;
  }

  template <typename F, typename P,
	    template <class> class I>
  void
  particle_container<F, P, I>::after_matching()
  {
    frame_cpt++;
  }

  template <typename F, typename P,
	    template <class> class I>
  void
  particle_container<F, P, I>::after_new_particles()
  {
    compact();
  }

  template <typename F, typename P,
	    template <class> class I>
  void
  particle_container<F, P, I>::compact()
  {
    SCOPE_PROF(compation);
    compact_has_run_ = true;

    matches_.resize(particles_vec_.size());
    std::fill(matches_.begin(), matches_.end(), -1);

    auto pts_it = particles_vec_.begin();
    auto feat_it = features_vec_.begin();
    auto pts_res = particles_vec_.begin();
    auto feat_res = features_vec_.begin();

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

  template <typename F, typename P,
	    template <class> class I>
  void
  particle_container<F, P, I>::add(i_int2 p, const typename F::value_type& feature)
  {
    P pt;
    pt.age = 1;
    pt.speed = i_int2(0,0);
    pt.pos = p;
    pt.prev_match_time = frame_cpt;
    pt.next_match_time = frame_cpt + 1;
    sparse_buffer_(p) = particles_vec_.size();
    particles_vec_.push_back(pt);
    features_vec_.push_back(feature);
  }

  template <typename F, typename P,
	    template <class> class I>
  void
  particle_container<F, P, I>::remove(int i)
  {
    P& p = particles_vec_[i];
    p.age = 0;
  }

  template <typename F, typename P,
	    template <class> class I>
  void
  particle_container<F, P, I>::remove(const i_short2& pos)
  {
    P& p = particles_vec_[sparse_buffer_(p.pos)];
    p.age = 0;
  }

  template <typename F, typename P,
	    template <class> class I>
  void
  particle_container<F, P, I>::move(unsigned i, particle_coords dst, const feature_type& f)
  {
    auto& p = particles_vec_[i];
    particle_coords src = p.pos;
    p.age++;
    i_int2 new_speed = dst - src;
    if (p.age > 1)
      p.acceleration = (new_speed - p.speed);
    else
      p.acceleration = i_int2(0,0);
    p.speed = i_float2(new_speed) / (frame_cpt - p.prev_match_time);
    p.pos = dst;
    p.prev_match_time = frame_cpt;
    float period = 10.f;
    i_float2 mesure = p.speed;
    if (norml2(mesure) > 0.f)
      period = std::min(10.f, 10.f / norml2(mesure));
    if (period < 1)
      period = 1;
    // p.next_match_time = frame_cpt + period;
    p.next_match_time = frame_cpt + 1;
    if ((p.speed.y + p.speed.x) > 0)
      features_vec_[i] = f;

    sparse_buffer_(dst) = i;
  }

  template <typename F, typename P,
	    template <class> class I>
  void
  particle_container<F, P, I>::touch(unsigned i)
  {
    auto& p = particles_vec_[i];
    particle_coords src = p.pos;
    p.age++;
    sparse_buffer_(p.pos) = i;
  }

  template <typename F, typename P,
	    template <class> class I>
  bool
  particle_container<F, P, I>::has(i_int2 p) const
  {
    unsigned int r = -1;
    return sparse_buffer_(p) != r;
  }


  template <typename F, typename P,
	    template <class> class I>
  void
  particle_container<F, P, I>::clear()
  {
    memset(sparse_buffer_, 255);
    particles_vec_.clear();
    features_vec_.clear();
    matches_.clear();
  }

  template <typename F, typename P,
	    template <class> class I>
  bool
  particle_container<F, P, I>::is_coherent()
  {
    // for (unsigned i = 0; i < dense_particles().size(); i++)
    // {
    //   i_short2 pos = dense_particles()[i].pos;
    //   if (dense_particles()[i].age > 0 && sparse_particles()(pos).vpos != i)
    //   {
    // 	std::cout << "i: " << i << "vpos: " << sparse_particles()(pos).vpos  << " pos:" << pos << " vpos.pos:" <<
    // 		  dense_particles()[sparse_particles()(pos).vpos].pos << std::endl;
    // 	std::cout << "p.age: " << dense_particles()[i].age <<
    // 	             "sp.age: " << sparse_particles()(pos).age << std::endl;

    // 	return false;
    //   }
    // }

    // bool ok = true;
    // mt_apply2d(sparse_particles().domain(), [this, &ok] (i_int2 p)
    // 	       {
    // 		 if (this->sparse_particles()(p).age == 0) return;
    // 		 int vpos = this->sparse_particles()(p).vpos;
    // 		 if (this->dense_particles()[vpos].pos != p)
    // 		 {
    // 		   std::cout << "vpos: " << vpos << " " << this->dense_particles()[vpos].pos << " != " <<  p << std::endl;
    // 		   ok = false;
    // 		 }
    // 	       });
    return true;
  }

}

#endif
