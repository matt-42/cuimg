#ifndef CUIMG_TRACKING_STRATEGIES_HPP_
# define CUIMG_TRACKING_STRATEGIES_HPP_

# include <cuimg/tracking2/tracking_strategies.h>
# include <cuimg/tracking2/gradient_descent_matcher.h>
# include <cuimg/tracking2/predictions.h>
# include <cuimg/tracking2/filter_spacial_incoherences.h>
# include <cuimg/tracking2/merge_trajectories.h>
# include <cuimg/tracking2/rigid_transform_estimator.h>
# include <cuimg/run_kernel.h>
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

    bc2s_fast_gradient_gpu::bc2s_fast_gradient_gpu(const obox2d& d)
      : super(d)
    {
    }

    void
    bc2s_fast_gradient_gpu::init()
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
      std::cout << "strategy_update start" << std::endl;

      feature_.update(in, architecture());


      match_particles(pset);


      estimate_camera_motion(pset);

      if (!(frame_cpt_ % detector_frequency_))
      {
        detector_.update(in);
	new_particles(pset);
      }

      pset.tick();
      frame_cpt_++;

      std::cout << "strategy_update done" << std::endl;

    }

    template<typename F, typename I, typename P>
    struct match_particles_kernel
    {
    private:
      typename P::kernel_type pset;
      typename kernel_type<F>::ret feature;
      typename I::kernel_type contrast;
      int frame_cpt;
      i_int2 camera_motion;
      i_int2 prev_camera_motion;
      int detector_frequency;

    public:
      match_particles_kernel(P& pset_, F& feature_, const I& contrast_, int frame_cpt_,
			     i_short2 camera_motion_, i_short2 prev_camera_motion_,
			     int detector_frequency_)
	: pset(pset_),
	  feature(feature_),
	  contrast(contrast_),
	  frame_cpt(frame_cpt_),
	  camera_motion(camera_motion_),
	  prev_camera_motion(prev_camera_motion_),
	  detector_frequency(detector_frequency_)
      {
      }

      inline __host__ __device__ void
      operator()(int i)
      {
	particle& part = pset.dense_particles()[i];
	if (part.age > 0
	    && part.next_match_time == frame_cpt)
	{
	  i_short2 pos = part.pos;
	  //i_short2 pred = multiscale_prediction(*this, p); // Prediction
	  i_short2 pred = motion_based_prediction(part, prev_camera_motion, camera_motion); // Prediction
	  float pos_distance = feature.distance(pset.features()[i], pos);
	  if ((feature.domain() - border(8)).has(pred))
	  {
	    float distance;
	    i_short2 match = two_step_gradient_descent_match(pred, pset.features()[i], feature, distance);
	    //i_short2 match = pred;
	    //i_short2 match = gradient_descent_match(pred, pset.features()[i], feature, distance);
	    if (feature.domain().has(match) //and detector_.saliency()(match) > 0.f
		and distance < 300 and part.fault < 10 //and pos_distance >= distance
		)
	    {
	      if (!(frame_cpt % detector_frequency) and contrast(match) <= 10.f) part.fault++;
	      if (!(frame_cpt % detector_frequency) and contrast(match) <= 1.f)
		pset.remove(i);
	      else
		//if (detector_.saliency()(match) <= 5.f) part.fault++;
		// if (distance > 300)
		// {
		// 	part.fault++;
		// 	match = pred;
		// }
		pset.move(i, match, feature(match));
	    }
	    else
	      pset.remove(i);
	  }
	  else
	    pset.remove(i);
	}
	else if (part.age > 0)
	  pset.touch(i);

	//assert(pset.dense_particles()[i].age == pset.sparse_particles()(part.pos).age);
      }
    };

    template<typename I, typename P>
    struct other_match_kernels
    {
      typename P::kernel_type pset;
      typename I::kernel_type flow;
      int flow_ratio;

      other_match_kernels(P& pset_, I& flow_, int flow_ratio_)
	: pset(pset_),
	  flow(flow_),
	  flow_ratio(flow_ratio_)
      {
      }


      inline __host__ __device__
      void merge_trajectories(int i)
      {
	particle& p = pset.dense_particles()[i];
	::cuimg::merge_trajectories(pset, p);
      }

      inline __host__ __device__
      void filter_bad_particles(int i)
      {
	particle& part = pset.dense_particles()[i];
	if (part.age > 0)
	{
	  if (is_spacial_incoherence(pset, part.pos))
	    pset.remove(i);
	}

      }
    };


    template<typename I, typename P>
    struct merge_trajectories_kernel
    {
      typename P::kernel_type pset;

      merge_trajectories_kernel(P& pset_)
	: pset(pset_)
      {}

      inline __host__ __device__
      void operator()(int i)
      {
	particle& p = pset.dense_particles()[i];
	::cuimg::merge_trajectories(pset, p);
      }
    };

    template<typename P>
    struct filter_bad_particle_kernel
    {
      typename P::kernel_type pset;

      filter_bad_particle_kernel(P& pset_)
	: pset(pset_)
      {}

      inline __host__ __device__
      void operator()(int i)
      {
	particle& part = pset.dense_particles()[i];
	if (part.age > 0)
	{
	  if (is_spacial_incoherence(pset, part.pos))
	    pset.remove(i);
	}
      }
    };

    template<typename F, typename D, typename P, typename I>
    void
    generic_strategy<F, D, P, I>::match_particles(particles_type& pset)
    {
      cudaThreadSynchronize();
      check_cuda_error();


      pset.before_matching();

      std::cout << "match_particles start" << std::endl;

      cudaThreadSynchronize();
      check_cuda_error();

      // Matching
      START_PROF(matcher);
      typename kernel_type<F>::ret feature_gpu = feature_;
      // int offset[8];
      // cudaMemcpyFromSymbol(offset, cuda_bc2s_offsets_s1, sizeof(offset));
      // for (unsigned i = 0; i < 8; i++)
      // 	std::cout << offset[i] << std::endl; 

      run_kernel1d_functor(match_particles_kernel<F, I, P>(pset, feature_, detector_.contrast(), frame_cpt_,
							   camera_motion(), prev_camera_motion(), detector_frequency_),
			   pset.dense_particles().size(),
			   typename particles_type::architecture());

      END_PROF(matcher);
      cudaThreadSynchronize();
      check_cuda_error();

      std::cout << "match_particles end" << std::endl;

      // Compute sparse flow.
#if 0
      START_PROF(sparse_flow);
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
      END_PROF(sparse_flow);
#endif

	cudaThreadSynchronize();
	check_cuda_error();

      if (!(frame_cpt_ % filtering_frequency_))
      {
	START_PROF(merge_trajectories);

	std::cout << "merge_trajectories" << std::endl;
	typedef other_match_kernels<flow_image_t, P> OM;
	merge_trajectories_call();

	run_kernel1d_functor(merge_trajectories<I, P>(pset),
			     pset.dense_particles().size(),
			     typename particles_type::architecture());

	run_kernel1d<OM, &OM::merge_trajectories>(OM(pset, flow_, flow_ratio),
						  pset.dense_particles().size(),
						  typename particles_type::architecture());
	END_PROF(merge_trajectories);

	cudaThreadSynchronize();
	check_cuda_error();

	// ****** Filter bad particles.

	std::cout << "filter_bad_particles" << std::endl;
	START_PROF(filter_spacial_incoherences);

	run_kernel1d<OM, &OM::filter_bad_particles> (OM(pset, flow_, flow_ratio),
						   pset.dense_particles().size(),
						   typename particles_type::architecture());

	cudaThreadSynchronize();
	check_cuda_error();

	std::cout << "filter_bad_particles end" << std::endl;
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
      camera_motion_ = dominant_speed_estimator_.estimate(pset, prev_camera_motion_, typename F::architecture());
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


  }


}

#endif

