#ifndef CUIMG_TRACKING_STRATEGIES_H_
# define CUIMG_TRACKING_STRATEGIES_H_

# include <cuimg/architectures.h>
# include <cuimg/gl.h>

# include <cuimg/cpu/host_image2d.h>

# include <cuimg/tracking2/bc2s_feature.h>
# include <cuimg/tracking2/mdfl_detector.h>
# include <cuimg/tracking2/particle_container.h>

namespace cuimg
{
  namespace tracking_strategies
  {

    struct bc2s_mdfl_gradient_cpu
    {
    public:
      typedef bc2s_feature<host_image2d> feature;
      typedef mdfl_1s_detector detector;
      typedef particle_container<feature> particles_type;
      typedef host_image2d<gl8u> input;

      template <typename T>
      static inline void init(T& tr);

      template <typename T>
      static inline i_short2 prediction(const particle& p, T& tr);

      template <typename T>
      static inline void match_particles(T& tr);

      template <typename T>
      static inline void new_particles(T& tr);

      template <typename T>
      static inline i_short2 estimate_camera_motion(T& tr);

    };

  }

}

# include <cuimg/tracking2/tracking_strategies.hpp>

#endif
