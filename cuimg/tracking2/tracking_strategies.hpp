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

    template <typename T>
    void
    bc2s_mdfl_gradient_cpu::init(T& tr)
    {
      tr.detector()
	.set_contrast_threshold(0.01f)
	.set_dev_threshold(0.5f);
    }

    template <typename T>
    void
    bc2s_mdfl_gradient_cpu::match_particles(T& tr)
    {
      auto& pset = tr.pset();
      pset.before_matching();

      START_PROF(matcher);
#pragma omp parallel for schedule(static, 300)
      for (unsigned i = 0; i < pset.dense_particles().size(); i++)
      {
        particle& part = pset.dense_particles()[i];
        if (part.age > 0)
        {
          i_short2 pos = part.pos;
          i_short2 pred = prediction(part, tr);
	  if ((tr.domain() - border(7)).has(pred))
	  {
	    i_short2 match = gradient_descent_match(pred, pset.features()[i], tr.feature());
	    if (tr.domain().has(match) and tr.detector().saliency()(match) > 0.f)
	      pset.move(i, match);
	    else
	      pset.remove(i);
	  }
	  else
	    pset.remove(i);
        }

	//assert(pset.dense_particles()[i].age == pset.sparse_particles()(part.pos).age);

      }
      END_PROF(matcher);

      // pset.after_matching();

      // for (unsigned i = 0; i < pset.dense_particles().size(); i++)
      // 	if (pset.dense_particles()[i].age > 0)
      // 	  merge_trajectories(pset, i);
      // for (unsigned i = 0; i < pset.dense_particles().size(); i++)
      // 	if (pset.dense_particles()[i].age > 0)
      // 	  merge_trajectories(pset, i);

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

      pset.compact();
    }


    template <typename T>
    inline void
    bc2s_mdfl_gradient_cpu::new_particles(T& tr)
    {
      tr.detector().new_particles(tr.feature(), tr.pset());
      tr.pset().after_new_particles();
    }

    template <typename T>
    inline i_short2
    bc2s_mdfl_gradient_cpu::estimate_camera_motion(T& tr)
    {
      //return estimate_dominent_motion(tr.pset());
      return i_short2(0,0);
    }

    template <typename T>
    inline i_short2
    bc2s_mdfl_gradient_cpu::prediction(const particle& p, T& tr)
    {
      return motion_based_prediction(p, tr.prev_camera_motion(), tr.camera_motion());
    }

  }

}

#endif

