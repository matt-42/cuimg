#ifndef CUIMG_TRACKING_STRATEGIES_H_
# define CUIMG_TRACKING_STRATEGIES_H_


# include <cuimg/tracking2/tracker_defines.h>
# include <cuimg/architectures.h>
# include <cuimg/gl.h>

# include <cuimg/cpu/host_image2d.h>

# include <cuimg/gl.h>

# include <cuimg/tracking2/bc2s_feature.h>
# include <cuimg/tracking2/mdfl_detector.h>
# include <cuimg/tracking2/miel2_detector.h>
# include <cuimg/tracking2/miel3_detector.h>
# include <cuimg/tracking2/fast_detector.h>
# include <cuimg/tracking2/fastnc_detector.h>
# include <cuimg/tracking2/dense_detector.h>
# include <cuimg/tracking2/regular_detector.h>
# include <cuimg/tracking2/gradient_detector.h>
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
      typedef typename feature_t::architecture::template image2d<std::pair<int, i_int2> >::ret flow_stats_t;
      typedef typename feature_t::architecture::template image2d<i_float2>::ret flow_t;
      typedef typename feature_t::architecture::template image2d<gl8u>::ret gl8u_image2d;
      typedef typename feature_t::architecture::template image2d<unsigned int>::ret uint_image2d;

      inline generic_strategy(const obox2d& o);

      inline void set_upper(self* s);
      inline self* upper();
      inline void init() {}

      inline self& set_detector_frequency(unsigned nframe);
      inline self& set_filtering_frequency(unsigned nframe);
      inline self& set_k(int k);

      inline i_short2 prediction(const particle& p);

      inline void match_particles(particles_type& pset);

      inline void new_particles(particles_type& pset);

      inline void estimate_camera_motion(const particles_type& pset);

      inline void update(const input& in, particles_type& pset);
      inline void filter(particles_type& pset);

      inline detector_t& detector() { return detector_; }
      inline feature_t& feature() { return feature_; }

      inline void clear();
      inline void create_detector_mask( particles_type& pset);

      inline i_int2 get_flow_at(const i_int2& p);
      inline const flow_t& flow() const { return flow_; }

      inline i_short2 camera_motion() const { return camera_motion_; }
      inline i_short2 prev_camera_motion() const { return prev_camera_motion_; }

      int border_needed() const;
      void set_with_merge(bool b);

    protected:
      feature_t feature_;
      feature_t prev_feature_;
      detector_t detector_;
      dominant_speed_estimator<typename feature_t::architecture> dominant_speed_estimator_;
      i_short2 camera_motion_;
      i_short2 prev_camera_motion_;
      self* upper_;
      self* lower_;
      int frame_cpt_;

      uint_image2d contrast_;
      gl8u_image2d mask_;
      gl8u_image2d new_points_map_;
      const int flow_ratio;
      flow_stats_t flow_stats_;
      flow_t flow_;
      uint_image2d multiscale_count_;

      int detector_frequency_;
      int filtering_frequency_;
      int k_;
      bool with_merge_;
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

    struct bc2s_miel2_gradient_cpu
      : public generic_strategy<bc2s_feature<cpu>, miel2_1s_detector,
				particle_container<bc2s_feature<cpu> >,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_feature<cpu>, miel2_1s_detector,
			       particle_container<bc2s_feature<cpu> >,
			       host_image2d<gl8u> > super;

      inline bc2s_miel2_gradient_cpu(const obox2d& o) : super(o) {}
      inline void init() {};
    };

    struct bc2s_miel3_gradient_cpu
      : public generic_strategy<bc2s_feature<cpu>, miel3_1s_detector,
				particle_container<bc2s_feature<cpu> >,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_feature<cpu>, miel3_1s_detector,
			       particle_container<bc2s_feature<cpu> >,
			       host_image2d<gl8u> > super;

      inline bc2s_miel3_gradient_cpu(const obox2d& o) : super(o) {}
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

    struct bc2s_fastnc_gradient_cpu
      : public generic_strategy<bc2s_feature<cpu>, fastnc_detector<cpu>,
				particle_container<bc2s_feature<cpu> >,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_feature<cpu>, fastnc_detector<cpu>,
			       particle_container<bc2s_feature<cpu> >,
			       host_image2d<gl8u> > super;

      inline bc2s_fastnc_gradient_cpu(const obox2d& o) :super(o) {}

      inline void init() {}
    };


    struct bc2s_regular_gradient_cpu
      : public generic_strategy<bc2s_feature<cpu>, regular_detector,
				particle_container<bc2s_feature<cpu> >,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_feature<cpu>, regular_detector,
			       particle_container<bc2s_feature<cpu> >,
			       host_image2d<gl8u> > super;

      inline bc2s_regular_gradient_cpu(const obox2d& o) :super(o) {}

      inline void init() {}
    };

    struct bc2s_gradient_gradient_cpu
      : public generic_strategy<bc2s_feature<cpu>, gradient_detector,
				particle_container<bc2s_feature<cpu> >,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_feature<cpu>, gradient_detector,
			       particle_container<bc2s_feature<cpu> >,
			       host_image2d<gl8u> > super;

      inline bc2s_gradient_gradient_cpu(const obox2d& o) :super(o) {}

      inline void init() {}
    };
    
    #ifndef NO_CUDA
    struct bc2s_fast_gradient_gpu
      : public generic_strategy<bc2s_feature<cuda_gpu>, fast_detector<cuda_gpu>,
				particle_container<bc2s_feature<cuda_gpu>, particle, cuda_gpu>,
				device_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_feature<cuda_gpu>, fast_detector<cuda_gpu>,
			       particle_container<bc2s_feature<cuda_gpu>, particle, cuda_gpu>,
			       device_image2d<gl8u> > super;

      inline bc2s_fast_gradient_gpu(const obox2d& o) : super(o) {}

      inline void init() {}
    };
    #endif
    
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

    struct bc2s_256_dense_gradient_cpu
      : public generic_strategy<bc2s_256_feature<cpu>, dense_detector,
				particle_container<bc2s_256_feature<cpu> >,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_256_feature<cpu>, dense_detector,
			       particle_container<bc2s_256_feature<cpu> >,
			       host_image2d<gl8u> > super;

      inline bc2s_256_dense_gradient_cpu(const obox2d& o) : super(o) {}
      inline void init() {}
    };


    struct bc2s_256_miel2_gradient_cpu
      : public generic_strategy<bc2s_256_feature<cpu>, miel2_1s_detector,
				particle_container<bc2s_256_feature<cpu> >,
				host_image2d<gl8u> >
    {
    public:
      typedef generic_strategy<bc2s_256_feature<cpu>, miel2_1s_detector,
			       particle_container<bc2s_256_feature<cpu> >,
			       host_image2d<gl8u> > super;

      inline bc2s_256_miel2_gradient_cpu(const obox2d& o) : super(o) {}
      inline void init() {}
    };

  }

}

# include <cuimg/tracking2/tracking_strategies.hpp>

#endif
