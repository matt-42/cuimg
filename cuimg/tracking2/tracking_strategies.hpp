#ifndef CUIMG_TRACKING_STRATEGIES_HPP_
# define CUIMG_TRACKING_STRATEGIES_HPP_

# include <cuimg/tracking2/tracking_strategies.h>
# include <cuimg/tracking2/gradient_descent_matcher.h>
# include <cuimg/tracking2/predictions.h>

namespace cuimg
{
  namespace tracking_strategies
  {

    template <typename T>
    void
    bc2s_mdfl_gradient_cpu::match_particles(T& tr)
    {
      auto& pset = tr.pset();
      pset.before_matching();

#pragma omp parallel for schedule(static, 100)
      for (unsigned i = 0; i < pset.dense_particles().size(); i++)
      {
	const particle_p& part = pset.dense_particles()[i];
	i_short2 pos = part.pos;
	i_short2 pred = prediction(part, tr);
	i_short2 match = gradient_descent_match(pred, pset.feature_at(pos), tr.feature());
	pset.move(pos, match);
      }

      // FIXME merge_trajectories(pset);
      pset.after_matching();
    }

    template <typename T>
    inline void
    bc2s_mdfl_gradient_cpu::new_particles(T& tr)
    {
      tr.detector().new_particles(tr.feature(), tr.pset(), 0.5f, 0.001f);
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
    bc2s_mdfl_gradient_cpu::prediction(const particle_p& p, T& tr)
    {
      return motion_based_prediction(p, tr.prev_camera_motion(), tr.camera_motion());
    }

  }

}

#endif
