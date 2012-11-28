#ifndef CUIMG_PARTICLE_CONTAINER_HP_
# define CUIMG_PARTICLE_CONTAINER_HP_

# include <cuimg/memset.h>
# include <cuimg/tracking2/compact_particles_image.h>
# include <cuimg/mt_apply.h>

namespace cuimg
{

  // ################### Particle container methods ############
  // ###########################################################

  template <typename F, typename P,
	    template <class> class I>
  particle_container<F, P, I>::particle_container(const obox2d& d)
    : sparse_buffer_(d)
  {
    particles_vec_.reserve((d.nrows() * d.ncols()) / 10);
    features_vec_.reserve((d.nrows() * d.ncols()) / 10);
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
  I<unsigned short>&
  particle_container<F, P, I>::sparse_particles()
  {
    return sparse_buffer_;
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
  const I<unsigned short>&
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

    // std::cout << particles_vec_.size() << std::endl;
    // std::cout << pts_res - particles_vec_.begin() << std::endl;
    particles_vec_.resize(pts_res - particles_vec_.begin());
    features_vec_.resize(feat_res - features_vec_.begin());

    //compact_particles_image(*cur_particles_img_, particles_vec_);
  }

  template <typename F, typename P,
	    template <class> class I>
  void
  particle_container<F, P, I>::add(i_int2 p, const typename F::value_type& feature)
  {
    particle pt;
    pt.age = 1;
    pt.speed = i_int2(0,0);
    pt.pos = p;

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
  particle_container<F, P, I>::move(unsigned i, i_int2 dst, const feature_type& f)
  {
    auto& p = particles_vec_[i];
    i_int2 src = p.pos;
    p.age++;
    i_int2 new_speed = dst - src;
    if (p.age > 1)
      p.acceleration = new_speed - p.speed;
    else
      p.acceleration = i_int2(0,0);
    p.speed = new_speed;
    p.pos = dst;

    if ((p.speed.y + p.speed.x) > 0)
      features_vec_[i] = f;

    sparse_buffer_(dst) = i;
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
