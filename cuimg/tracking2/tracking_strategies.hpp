#ifndef CUIMG_TRACKING_STRATEGIES_HPP_
# define CUIMG_TRACKING_STRATEGIES_HPP_

# include <cuimg/tracking2/tracking_strategies.h>
# include <cuimg/tracking2/gradient_descent_matcher.h>
# include <cuimg/tracking2/predictions.h>
# include <cuimg/tracking2/filter_spacial_incoherences.h>
# include <cuimg/tracking2/merge_trajectories.h>

namespace cuimg
{
  namespace tracking_strategies
  {

    bc2s_mdfl_gradient_cpu::bc2s_mdfl_gradient_cpu(const obox2d& d)
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
      detector_
	.set_contrast_threshold(0)
	.set_fast_threshold(60);
    }

    template<typename F, typename D, typename P, typename I>
    generic_strategy<F, D, P, I>::generic_strategy(const obox2d& d)
      : feature_(d),
	detector_(d),
	dominant_speed_estimator_(d),
	camera_motion_(0,0),
	prev_camera_motion_(0,0),
	upper_(0),
	frame_cpt_(0)
    {
    }

    template<typename F, typename D, typename P, typename I>
    void
    generic_strategy<F, D, P, I>::update(const I& in, particles_type& pset)
    {
      feature_.update(in);
      match_particles(pset);

      estimate_camera_motion(pset);

      if (!(frame_cpt_ % 5))
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
	    i_short2 match = gradient_descent_match(pred, pset.features()[i], feature_, distance);
	    if (feature_.domain().has(match) //and detector_.saliency()(match) > 0.f
		and distance < 300 and part.fault < 10 //and pos_distance >= distance
		)
	    {
	      if (detector_.saliency()(match) <= 5.f) part.fault++;
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


      //if (false)
      if (!(frame_cpt_ % 5))
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
	  if (part.age > 0 and is_spacial_incoherence(pset, part.pos))
	    pset.remove(i);
	}

	END_PROF(filter_spacial_incoherences);
      }
    }

    template<typename F, typename D, typename P, typename I>
    inline void
    generic_strategy<F, D, P, I>::new_particles(particles_type& pset)
    {
      detector_.new_particles(feature_, pset);
      pset.compact();
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
      if (upper_)
	return motion_based_prediction(p, upper_->prev_camera_motion_*2, upper_->camera_motion_*2);
      else
	return motion_based_prediction(p);
    }

    template<typename F, typename D, typename P, typename I>
    void
    generic_strategy<F, D, P, I>::set_upper(self* u)
    {
      u->lower_ = this;
      upper_ = u;
    }


  }

}

#endif

