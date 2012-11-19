#ifndef CUIMG_TRACKING_STRATEGIES_H_
# define CUIMG_TRACKING_STRATEGIES_H_

# include <cuimg/architectures.h>
# include <cuimg/gl.h>

# include <cuimg/cpu/host_image2d.h>

# include <cuimg/tracking2/bc2s_feature.h>
# include <cuimg/tracking2/mdfl_detector.h>
# include <cuimg/tracking2/particle_container.h>
# include <cuimg/tracking2/dominant_speed_estimator.h>

namespace cuimg
{
  namespace tracking_strategies
  {

    struct bc2s_mdfl_gradient_cpu
    {
    public:
      typedef bc2s_mdfl_gradient_cpu self;
      typedef bc2s_feature<host_image2d> feature_t;
      typedef mdfl_1s_detector detector_t;
      typedef particle_container<feature_t> particles_type;
      typedef host_image2d<gl8u> input;

      inline bc2s_mdfl_gradient_cpu(const obox2d& o);

      inline void set_upper(self* s);
      inline void init();

      inline i_short2 prediction(const particle& p);

      inline void match_particles(particles_type& pset);

      inline void new_particles(particles_type& pset);

      inline void estimate_camera_motion(const particles_type& pset);

      template <typename I>
      inline void update(const I& in, particles_type& pset);

    private:
      feature_t feature_;
      detector_t detector_;
      dominant_speed_estimator dominant_speed_estimator_;
      i_short2 camera_motion_;
      i_short2 prev_camera_motion_;
      self* upper_;
      self* lower_;
      int frame_cpt_;
    };

  }

}

# include <cuimg/tracking2/tracking_strategies.hpp>

#endif
