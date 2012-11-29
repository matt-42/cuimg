#ifndef CUIMG_TRACKING_STRATEGIES_H_
# define CUIMG_TRACKING_STRATEGIES_H_

# include <cuimg/architectures.h>
# include <cuimg/gl.h>

# include <cuimg/cpu/host_image2d.h>

# include <cuimg/tracking2/bc2s_feature.h>
# include <cuimg/tracking2/mdfl_detector.h>
# include <cuimg/tracking2/fast_detector.h>
# include <cuimg/tracking2/dense_detector.h>
# include <cuimg/tracking2/particle_container.h>
# include <cuimg/tracking2/dominant_speed_estimator.h>

namespace cuimg
{
  namespace tracking_strategies
  {

    template<typename FEATURE,
	     typename DETECTOR,
	     typename PARTICLE_SET,
	     typename INPUT>
    struct generic_strategy
    {
    public:
      typedef generic_strategy<FEATURE, DETECTOR, PARTICLE_SET, INPUT>  self;
      typedef FEATURE feature_t;
      typedef DETECTOR detector_t;
      typedef PARTICLE_SET particles_type;
      typedef INPUT input;

      inline generic_strategy(const obox2d& o);

      inline void set_upper(self* s);
      inline void init();

      inline i_short2 prediction(const particle& p);

      inline void match_particles(particles_type& pset);

      inline void new_particles(particles_type& pset);

      inline void estimate_camera_motion(const particles_type& pset);

      inline void update(const input& in, particles_type& pset);

      inline detector_t& detector() { return detector_; }

    protected:
      feature_t feature_;
      detector_t detector_;
      dominant_speed_estimator dominant_speed_estimator_;
      i_short2 camera_motion_;
      i_short2 prev_camera_motion_;
      self* upper_;
      self* lower_;
      int frame_cpt_;
    };

    struct bc2s_mdfl_gradient_cpu
      : public generic_strategy<bc2s_feature<host_image2d>, mdfl_1s_detector,
				particle_container<bc2s_feature<host_image2d> >,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_feature<host_image2d>, mdfl_1s_detector,
			       particle_container<bc2s_feature<host_image2d> >,
			       host_image2d<gl8u> > super;

      inline bc2s_mdfl_gradient_cpu(const obox2d& o);

      inline void init();
    };

    struct bc2s_mdfl_gradient_multiscale_prediction_cpu
      : public bc2s_mdfl_gradient_cpu
    {
    public:

      inline bc2s_mdfl_gradient_multiscale_prediction_cpu(const obox2d& d)
	: bc2s_mdfl_gradient_cpu(d), flow_(d / 8) {}

      inline i_short2 prediction(const particle& p);
      inline void match_particles(particles_type& pset);

      inline const host_image2d<std::pair<int, i_float2> >& flow() const { return flow_; }
    private:
      const static unsigned flow_ratio = 8;
      host_image2d<std::pair<int, i_float2> > flow_;
    };

    struct bc2s_fast_gradient_cpu
      : public generic_strategy<bc2s_feature<host_image2d>, fast_detector,
				particle_container<bc2s_feature<host_image2d> >,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_feature<host_image2d>, fast_detector,
			       particle_container<bc2s_feature<host_image2d> >,
			       host_image2d<gl8u> > super;

      inline bc2s_fast_gradient_cpu(const obox2d& o);

      inline void init();
    };


    struct bc2s_dense_gradient_cpu
      : public generic_strategy<bc2s_feature<host_image2d>, dense_detector,
				particle_container<bc2s_feature<host_image2d> >,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_feature<host_image2d>, dense_detector,
			       particle_container<bc2s_feature<host_image2d> >,
			       host_image2d<gl8u> > super;

      inline bc2s_dense_gradient_cpu(const obox2d& o) : super(o) {}
      inline void init() {}
    };

  }

}

# include <cuimg/tracking2/tracking_strategies.hpp>

#endif
