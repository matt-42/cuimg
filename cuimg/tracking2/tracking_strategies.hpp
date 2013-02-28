#ifndef CUIMG_TRACKING_STRATEGIES_HPP_
# define CUIMG_TRACKING_STRATEGIES_HPP_

# include <cuimg/tracking2/tracking_strategies.h>
# include <cuimg/tracking2/gradient_descent_matcher.h>
# include <cuimg/tracking2/predictions.h>
# include <cuimg/tracking2/filter_spacial_incoherences.h>
# include <cuimg/tracking2/merge_trajectories.h>
# include <cuimg/tracking2/rigid_transform_estimator.h>
# include <cuimg/iterate.h>

namespace cuimg
{
  namespace tracking_strategies
  {

    bc2s_mdfl_gradient_cpu::bc2s_mdfl_gradient_cpu(const obox2d& d)
      : super(d)
    {
    }

    bc2s64_mdfl_gradient_cpu::bc2s64_mdfl_gradient_cpu(const obox2d& d)
      : super(d)
    {
    }

    void
    bc2s_mdfl_gradient_cpu::init()
    {
      detector_
        .set_contrast_threshold(60)
        .set_dev_threshold(0.5f);
    }


    bc2s_fast_gradient_cpu::bc2s_fast_gradient_cpu(const obox2d& d)
      : super(d)
    {
    }

    void
    bc2s_fast_gradient_cpu::init()
    {
    }

    template<typename F, typename D, typename P, typename I>
    generic_strategy<F, D, P, I>::generic_strategy(const obox2d& d)
      : feature_(d),
	flow_ratio(8),
        detector_(d),
        dominant_speed_estimator_(d),
        camera_motion_(0,0),
        prev_camera_motion_(0,0),
        upper_(0),
        frame_cpt_(0),
	flow_(d / flow_ratio),
        detector_frequency_(1),
        filtering_frequency_(1)
    {
    }

    template<typename F, typename D, typename P, typename I>
    generic_strategy<F, D, P, I>&
    generic_strategy<F, D, P, I>::set_detector_frequency(unsigned nframe)
    {
      detector_frequency_ = nframe;
      if (upper_)
	upper_->set_detector_frequency(nframe);
      return *this;
    }

    template<typename F, typename D, typename P, typename I>
    generic_strategy<F, D, P, I>&
    generic_strategy<F, D, P, I>::set_filtering_frequency(unsigned nframe)
    {
      filtering_frequency_ = nframe;
      if (upper_)
	upper_->set_filtering_frequency(nframe);
      return *this;
    }

    template<typename F, typename D, typename P, typename I>
    void
    generic_strategy<F, D, P, I>::update(const I& in, particles_type& pset)
    {
      feature_.update(in);
      match_particles(pset);

      estimate_camera_motion(pset);

      if (!(frame_cpt_ % detector_frequency_))
      {
        detector_.update(in);
        new_particles(pset);
      }

      frame_cpt_++;
    }

    template<typename F, typename D, typename P, typename I>
    void
    generic_strategy<F, D, P, I>::match_particles(particles_type& pset)
    {
      pset.before_matching();

      START_PROF(matcher);
#pragma omp parallel for schedule(static, 300)
      for (unsigned i = 0; i < pset.dense_particles().size(); i++)
      {
        particle& part = pset.dense_particles()[i];
        if (part.age > 0)
        {
          i_short2 pos = part.pos;
          i_short2 pred = prediction(part);
	  float pos_distance = feature_.distance(pset.features()[i], pos);
	  if ((feature_.domain() - border(7)).has(pred))
	  {
	    float distance;
	    i_short2 match = two_step_gradient_descent_match(pred, pset.features()[i], feature_, distance);
	    //i_short2 match = gradient_descent_match(pred, pset.features()[i], feature_, distance);
	    if (feature_.domain().has(match) //and detector_.saliency()(match) > 0.f
		and distance < 300 and part.fault < 10 //and pos_distance >= distance
		)
	    {
	      // if (!(frame_cpt_ % detector_frequency_) and detector_.contrast()(match) <= 10.f) part.fault++;
	      // if (!(frame_cpt_ % detector_frequency_) and detector_.contrast()(match) <= 1.f)
	      // 	pset.remove(i);
	      // else
	      //if (detector_.saliency()(match) <= 5.f) part.fault++;
	      // if (distance > 300)
	      // {
	      // 	part.fault++;
	      // 	match = pred;
	      // }
	      pset.move(i, match, feature_(match));
	    }
	    else
	      pset.remove(i);
	  }
	  else
	    pset.remove(i);
        }

	//assert(pset.dense_particles()[i].age == pset.sparse_particles()(part.pos).age);

      }

      END_PROF(matcher);

      memset(flow_, 0);
      pset.for_each_particle_st
      	([this] (const particle& p)
      	 {
	   if (p.age > 1)
	   {
	     int r = flow_ratio;
	     i_int2 bin = p.pos / r;
	     flow_(bin).first++;
	     flow_(bin).second += p.speed;
	   }
      	 });

      [&] (i_int2 p) {
	if (flow_(p).first)
	  flow_(p).second /= flow_(p).first;
      } >> iterate(flow_.domain());

      if (!(frame_cpt_ % filtering_frequency_))
      {
	START_PROF(merge_trajectories);
	pset.for_each_particle_st([&pset] (particle& p) { merge_trajectories(pset, p); });
	END_PROF(merge_trajectories);

	// ****** Filter bad particles.

	START_PROF(filter_spacial_incoherences);
#pragma omp parallel for schedule(static, 300)
	for (unsigned i = 0; i < pset.dense_particles().size(); i++)
	{
	  particle& part = pset[i];
	  if (part.age > 0)
	  {
	    if (is_spacial_incoherence(pset, part.pos))
	      pset.remove(i);
	  }
	}

	END_PROF(filter_spacial_incoherences);
      }
    }

    template<typename F, typename D, typename P, typename I>
    inline void
    generic_strategy<F, D, P, I>::new_particles(particles_type& pset)
    {
      detector_.new_particles(feature_, pset);
      pset.after_new_particles();
    }

    template<typename F, typename D, typename P, typename I>
    inline void
    generic_strategy<F, D, P, I>::estimate_camera_motion(const particles_type& pset)
    {
      prev_camera_motion_ = camera_motion_;
      camera_motion_ = dominant_speed_estimator_.estimate(pset, prev_camera_motion_);
    }

    template<typename F, typename D, typename P, typename I>
    inline i_short2
    generic_strategy<F, D, P, I>::prediction(const particle& p)
    {
      // if (upper_)
      // 	return motion_based_prediction(p, upper_->prev_camera_motion_*2, upper_->camera_motion_*2);
      // else
      // 	return motion_based_prediction(p);
      //motion_based_prediction(*this, p);
      return multiscale_prediction(*this, p);
    }

    template<typename F, typename D, typename P, typename I>
    void
    generic_strategy<F, D, P, I>::set_upper(self* u)
    {
      u->lower_ = this;
      upper_ = u;
    }

    template<typename F, typename D, typename P, typename I>
    generic_strategy<F, D, P, I>*
    generic_strategy<F, D, P, I>::upper()
    {
      return upper_;
    }

    template<typename F, typename D, typename P, typename I>
    void
    generic_strategy<F, D, P, I>::clear()
    {
      frame_cpt_ = 0;
    }

    template<typename F, typename D, typename P, typename I>
    i_int2
      generic_strategy<F, D, P, I>::get_flow_at(const i_int2& p)
    {
      if (flow_(p / flow_ratio).first)
	return flow_(p / flow_ratio).second;
      else
      {
	particle part;
	part.age = 1;
	part.pos = p;
	i_short2 pred = prediction(part);
	if ((feature_.domain() - border(7)).has(pred))
	{
 	  float distance;
	  i_short2 match = two_step_gradient_descent_match(pred, feature_(p), feature_, distance);
	  if (feature_.domain().has(match))
	  {
	    flow_(p / flow_ratio).first = 1;
	    flow_(p / flow_ratio).second = match - p;
	    return match - p;
	  }
	}
      }

      return i_int2(0,0);
    }


    i_short2
    bc2s_mdfl_gradient_multiscale_prediction_cpu::prediction(const particle& p)
    {
      return multiscale_prediction(*this, p);
    }

    void
    bc2s_mdfl_gradient_multiscale_prediction_cpu::update(const host_image2d<gl8u>& in, particles_type& pset)
    {
      feature_.update(in);
      match_particles(pset);

      estimate_camera_motion(pset);

      if (!(frame_cpt_ % detector_frequency_))
      {
        detector_.update(in);
        new_particles(pset);
      }

      frame_cpt_++;
    }

    void
    bc2s_mdfl_gradient_multiscale_prediction_cpu::match_particles(particles_type& pset)
    {
      bc2s_mdfl_gradient_multiscale_prediction_cpu* upper =
	static_cast<bc2s_mdfl_gradient_multiscale_prediction_cpu*>(upper_);
      pset.before_matching();

      START_PROF(matcher);
#pragma omp parallel for schedule(static, 300)
      for (unsigned i = 0; i < pset.dense_particles().size(); i++)
      {
        particle& part = pset.dense_particles()[i];
        if (part.age > 0)
        {
          i_short2 pos = part.pos;
          i_short2 pred = prediction(part);
	  float pos_distance = feature_.distance(pset.features()[i], pos, 2);
	  if ((feature_.domain() - border(7)).has(pred))
	  {
	    float distance;
	    i_short2 match = two_step_gradient_descent_match(pred, feature_(pos), feature_, distance);
	    //i_short2 match = gradient_descent_match(pred, feature_(pos), feature_, distance, 1);
	    if (feature_.domain().has(match) //and detector_.saliency()(match) > 0.f
		and part.fault < 5 //and pos_distance >= distance
		and distance < 300
		)
	    {
	      if (detector_.contrast()(match) <= 10.f) part.fault++;
	      if (detector_.contrast()(match) <= 5.f)
	      	pset.remove(i);
	      else
		pset.move(i, match, feature_(match));
	    }
	    else
	      pset.remove(i);
	  }
	  else
	    pset.remove(i);
        }

	//assert(pset.dense_particles()[i].age == pset.sparse_particles()(part.pos).age);

      }
      END_PROF(matcher);


      memset(flow_, 0);
      pset.for_each_particle_st
      	([this] (const particle& p)
      	 {
	   if (p.age > 1)
	   {
	     i_int2 bin = p.pos / flow_ratio;
	     flow_(bin).first++;
	     flow_(bin).second += p.speed;
	   }
      	 });

      [&] (i_int2 p) {
	if (flow_(p).first)
	  flow_(p).second /= flow_(p).first;
      } >> iterate(flow_.domain());

      if (!(frame_cpt_ % filtering_frequency_))
      {
	START_PROF(merge_trajectories);
	pset.for_each_particle_st([&pset] (particle& p) { merge_trajectories(pset, p); });
	END_PROF(merge_trajectories);

	// ****** Filter bad particles.

	START_PROF(filter_spacial_incoherences);
#pragma omp parallel for schedule(static, 300)
	for (unsigned i = 0; i < pset.dense_particles().size(); i++)
	{
	  particle& part = pset[i];
	  if (part.age > 0)
	  {
	    if (is_spacial_incoherence(pset, part.pos))
	      pset.remove(i);
	    else
	    {
	      if (upper)
	      {
	      	auto f = upper->flow_(part.pos / (2*flow_ratio));
	      	if (f.first > 2 and norml2(part.speed - 2 * f.second) > 5.f)
	      	  pset.remove(i);
	      }
	    }
	  }
	}

	END_PROF(filter_spacial_incoherences);
      }

    }

  }


}

#endif

