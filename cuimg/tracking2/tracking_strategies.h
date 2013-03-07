#ifndef CUIMG_TRACKING_STRATEGIES_H_
# define CUIMG_TRACKING_STRATEGIES_H_

# include <cuimg/architectures.h>
# include <cuimg/gl.h>

# include <cuimg/cpu/host_image2d.h>

# include <cuimg/gl.h>

# include <cuimg/tracking2/bc2s_feature.h>
# include <cuimg/tracking2/mdfl_detector.h>
# include <cuimg/tracking2/fast_detector.h>
# include <cuimg/tracking2/dense_detector.h>
# include <cuimg/tracking2/particle_container.h>
# include <cuimg/tracking2/dominant_speed_estimator.h>
# include <cuimg/tracking2/rigid_transform_estimator.h>

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
      typedef typename particles_type::particle_type particle_type;
      typedef INPUT input;
      typedef typename particles_type::architecture architecture;
      inline generic_strategy(const obox2d& o);

      inline void set_upper(self* s);
      inline self* upper();
      inline void init() {}

      inline self& set_detector_frequency(unsigned nframe);
      inline self& set_filtering_frequency(unsigned nframe);

      inline i_short2 prediction(const particle& p);

      inline void match_particles(particles_type& pset);

      inline void new_particles(particles_type& pset);

      inline void estimate_camera_motion(const particles_type& pset);

      inline void update(const input& in, particles_type& pset);

      inline detector_t& detector() { return detector_; }
      inline feature_t& feature() { return feature_; }

      inline void clear();

      inline i_int2 get_flow_at(const i_int2& p);

      inline const i_short2 camera_motion() const { return camera_motion_; }
      inline const i_short2 prev_camera_motion() const { return prev_camera_motion_; }

    protected:
      feature_t feature_;
      detector_t detector_;
      dominant_speed_estimator<typename feature_t::architecture> dominant_speed_estimator_;
      i_short2 camera_motion_;
      i_short2 prev_camera_motion_;
      self* upper_;
      self* lower_;
      int frame_cpt_;

      const int flow_ratio;
      typedef typename feature_t::architecture::template image2d<std::pair<int, i_float2> >::ret flow_image_t;
      flow_image_t flow_;

      int detector_frequency_;
      int filtering_frequency_;
    };

    struct bc2s_mdfl_gradient_cpu
      : public generic_strategy<bc2s_feature<cpu>, mdfl_1s_detector,
    				particle_container<bc2s_feature<cpu>, particle, cpu>,
    				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_feature<cpu>, mdfl_1s_detector,
    			       particle_container<bc2s_feature<cpu>, particle, cpu>,
    			       host_image2d<gl8u> > super;

      inline bc2s_mdfl_gradient_cpu(const obox2d& o);

      inline void init();
    };


    /* struct bc2s_mdfl_gradient_cpu */
    /*   : public generic_strategy<bc2s_256_feature<host_image2d>, mdfl_1s_detector, */
    /* 				particle_container<bc2s_256_feature<host_image2d> >, */
    /* 				host_image2d<gl8u> > */
    /* { */
    /* public: */
    /*   typedef generic_strategy<bc2s_256_feature<host_image2d>, mdfl_1s_detector, */
    /* 			       particle_container<bc2s_256_feature<host_image2d> >, */
    /* 			       host_image2d<gl8u> > super; */

    /*   inline bc2s_mdfl_gradient_cpu(const obox2d& o); */

    /*   inline void init(); */
    /* }; */


    struct bc2s64_mdfl_gradient_cpu
      : public generic_strategy<bc2s64_feature<cpu>, mdfl_1s_detector,
				particle_container<bc2s64_feature<cpu> >,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s64_feature<cpu>, mdfl_1s_detector,
			       particle_container<bc2s64_feature<cpu> >,
			       host_image2d<gl8u> > super;

      inline bc2s64_mdfl_gradient_cpu(const obox2d& o);
    };


    struct bc2s_mdfl2s_gradient_cpu
      : public generic_strategy<bc2s_feature<cpu>, mdfl_2s_detector,
				particle_container<bc2s_feature<cpu> >,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_feature<cpu>, mdfl_2s_detector,
			       particle_container<bc2s_feature<cpu> >,
			       host_image2d<gl8u> > super;

      inline bc2s_mdfl2s_gradient_cpu(const obox2d& o) : super(o) {}
      inline void init() {};
    };

    struct bc2s_mdfl_gradient_multiscale_prediction_cpu
      : public bc2s_mdfl_gradient_cpu
    {
    public:

      inline bc2s_mdfl_gradient_multiscale_prediction_cpu(const obox2d& d)
	: bc2s_mdfl_gradient_cpu(d) {}

      inline i_short2 prediction(const particle& p);
      inline void match_particles(particles_type& pset);
      inline void update(const input& in, particles_type& pset);

      inline const host_image2d<std::pair<int, i_float2> >& flow() const { return flow_; }

    private:
    };

    struct bc2s_fast_gradient_cpu
      : public generic_strategy<bc2s_feature<cpu>, fast_detector<cpu>,
				particle_container<bc2s_feature<cpu>, particle, cpu>,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_feature<cpu>, fast_detector<cpu>,
			       particle_container<bc2s_feature<cpu>, particle, cpu>,
			       host_image2d<gl8u> > super;

      inline bc2s_fast_gradient_cpu(const obox2d& o);

      inline void init();
    };

    struct bc2s_fast_gradient_multiscale_prediction_cpu
      : public generic_strategy<bc2s_feature<cpu>, fast_detector<cpu>,
				particle_container<bc2s_feature<cpu> >,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_feature<cpu>, fast_detector<cpu>,
			       particle_container<bc2s_feature<cpu> >,
			       host_image2d<gl8u> > super;

      inline bc2s_fast_gradient_multiscale_prediction_cpu(const obox2d& o);

      inline void init();
    };

    struct bc2s_dense_gradient_cpu
      : public generic_strategy<bc2s_feature<cpu>, dense_detector,
				particle_container<bc2s_feature<cpu> >,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_feature<cpu>, dense_detector,
			       particle_container<bc2s_feature<cpu> >,
			       host_image2d<gl8u> > super;

      inline bc2s_dense_gradient_cpu(const obox2d& o) : super(o) {}
      inline void init() {}
    };

  }

}

# include <cuimg/tracking2/tracking_strategies.hpp>

#endif
