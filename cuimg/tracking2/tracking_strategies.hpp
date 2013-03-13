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
	contrast_(d),
	mask_(d),
	flow_stats_(d / flow_ratio),
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

    template<typename I, typename J>
    struct contrast_kernel
    {
      typename I::kernel_type input_;
      typename I::kernel_type out_;

      contrast_kernel(const I& input, J& output)
	: input_(input),
	  out_(output)
      {}

      inline __host__ __device__
      void operator()(i_int2 p)
      {
	const int d = 5;
	int res = 0;
	if ((input_.domain() - border(d)).has(p))
	  res = ::abs(int(input_(p + i_int2(0,d))) - int(input_(p + i_int2(0,-d)))) +
	    ::abs(int(input_(p + i_int2(-d,0))) - int(input_(p + i_int2(d,0))));
	out_(p) = res;
      }
    };

    template<typename F, typename D, typename P, typename I>
    void
    generic_strategy<F, D, P, I>::update(const I& in, particles_type& pset)
    {
      run_kernel2d_functor(contrast_kernel<I, gl8u_image2d>(in, contrast_),
			   contrast_.domain(), architecture());

      feature_.update(in, architecture());
      match_particles(pset);
      estimate_camera_motion(pset);

      if (!(frame_cpt_ % detector_frequency_))
      {
	memset(mask_, 0);
        detector_.update(in, mask_);
	new_particles(pset);
      }

      pset.tick();
      frame_cpt_++;
    }

    template<typename F, typename I, typename J, typename P>
    struct match_particles_kernel
    {
    private:
      typename P::kernel_type pset;
      typename kernel_type<F>::ret feature;
      typename I::kernel_type contrast;
      typename J::kernel_type upper_flow;
      int frame_cpt;
      i_int2 u_camera_motion;
      i_int2 u_prev_camera_motion;
      int detector_frequency;
      int flow_ratio;

    public:
      match_particles_kernel(P& pset_, F& feature_, const I& contrast_, const J& upper_flow_, int frame_cpt_,
			     i_short2 u_camera_motion_, i_short2 u_prev_camera_motion_,
			     int detector_frequency_, int flow_ratio_)
	: pset(pset_),
	  feature(feature_),
	  contrast(contrast_),
	  upper_flow(upper_flow_),
	  frame_cpt(frame_cpt_),
	  u_camera_motion(u_camera_motion_),
	  u_prev_camera_motion(u_prev_camera_motion_),
	  detector_frequency(detector_frequency_),
	  flow_ratio(flow_ratio_)
      {
      }

      inline __host__ __device__ void
      operator()(int i)
      {
	assert(i >= 0 && i < pset.size());
	particle& part = pset.dense_particles()[i];
	assert(pset.domain().has(part.pos));
	box2d domain = pset.domain() - border(6);
	assert(domain.has(part.pos));
	if (part.age > 0
	    //&& part.next_match_time == frame_cpt
	    )
	{
	  // Prediction.
	  i_short2 pred;
	  pred = multiscale_prediction(part, upper_flow, flow_ratio, u_prev_camera_motion, u_camera_motion);
	  //pred = motion_based_prediction(part, u_prev_camera_motion, u_camera_motion);

	  // Matching.
	  float pos_distance = feature.distance(pset.features()[i], part.pos);
	  if (domain.has(pred))
	  {
	    float distance;
	    std::pair<i_short2, float> match_res = two_step_gradient_descent_match(pred, pset.features()[i], feature);
	    i_short2 match = match_res.first;
	    distance = match_res.second;
	    if (domain.has(match)
		and distance < 300
		and part.fault < 10
		)
	    {
	      if (contrast(match) <= 10.f) part.fault++;
	      if (contrast(match) <= 1.f)
		pset.remove(i);
	      else
	      {
	        pset.move(i, match, feature(match));
		assert(pset.has(match));
		assert(pset.dense_particles()[i].age > 0);
	      }
	    }
	    else
	      pset.remove(i);
	  }
	  else
	    pset.remove(i);
	}
	else if (part.age > 0)
	  pset.touch(i);
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


    template<typename P>
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
    struct filter_bad_particles_kernel
    {
      typename P::kernel_type pset;

      filter_bad_particles_kernel(P& pset_)
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

    template<typename P, typename I>
    struct compute_flow_stats_kernel
    {
      typename P::kernel_type pset_;
      typename I::kernel_type stats_;
      int flow_ratio_;

      compute_flow_stats_kernel(P& pset, I& stats, int flow_ratio)
	: pset_(pset),
	  stats_(stats),
	  flow_ratio_(flow_ratio)
      {}

      inline __host__ __device__
      void operator()(i_int2 p)
      {
	unsigned count = 0;
	i_int2 sum(0,0);
	for (unsigned r = 0; r < flow_ratio_; r++)
	  for (unsigned c = 0; c < flow_ratio_; c++)
	  {
	    i_int2 n = p * flow_ratio_ + i_int2(r, c);
	    if (pset_.domain().has(n) && pset_.has(n) &&
		pset_(n).age > 1)
	    {
	      count++;
	      sum += pset_(n).speed;
	    }
	  }

	stats_(p).first = count;
	stats_(p).second = sum;
      }
    };

    template<typename I, typename J>
    struct flow_fusion_kernel
    {
      typename I::kernel_type stats_, upper_stats_;
      typename J::kernel_type flow_, upper_flow_;

      flow_fusion_kernel(I& stats, I& upper_stats,
			 J& flow, J& upper_flow)
	: stats_(stats),
	  upper_stats_(upper_stats),
	  flow_(flow),
	  upper_flow_(upper_flow)
      {}

      inline __host__ __device__
      void operator()(i_int2 p)
      {
	if (!stats_.has(p)) return;
	i_int2 bin = p;
	i_int2 ubin = bin / 2;

	if (stats_(bin).first > 1)
	  flow_(bin) = stats_(bin).second / stats_(bin).first;
	else
	  //if (upper_stats_(ubin).first > 1)
	  flow_(bin) = upper_flow_(ubin) * 2;
	//else flow_(bin) = i_float2(0.f, 0.f);
      }
    };

    template<typename I, typename J>
    struct flow_fusion_kernel_root
    {
      typename I::kernel_type stats_;
      typename J::kernel_type flow_;

      flow_fusion_kernel_root(I& stats, J& flow)
	: stats_(stats),
	  flow_(flow)
      {}

      inline __host__ __device__
      void operator()(i_int2 p)
      {
	if (!flow_.has(p)) return;
	i_int2 bin = p;
	i_int2 ubin = bin / 2;

	if (stats_(bin).first > 1)
	  flow_(bin) = stats_(bin).second / stats_(bin).first;
	else
	  flow_(bin) = i_float2(0.f, 0.f);
      }
    };

    template<typename F, typename D, typename P, typename I>
    void
    generic_strategy<F, D, P, I>::match_particles(particles_type& pset)
    {
      pset.before_matching();

      // Matching
      START_PROF(matcher);
      typename kernel_type<F>::ret feature_gpu = feature_;

      feature_.bind();

      i_short2 ucm = upper_ ? upper_->camera_motion_ : i_short2(0,0);
      i_short2 upcm = upper_ ? upper_->prev_camera_motion_ : i_short2(0,0);
      flow_t uf = upper_ ? upper_->flow_ : flow_t();
      match_particles_kernel<F, I, flow_t, P> func
	(pset, feature_, contrast_, uf, frame_cpt_,
	 ucm, upcm, detector_frequency_, flow_ratio);
      run_kernel1d_functor(func,
      			   pset.dense_particles().size(),
      			   typename particles_type::architecture());

      // cudaUnbindTexture(bc2s_tex_s1);
      // cudaUnbindTexture(bc2s_tex_s2);
      END_PROF(matcher);

      // Compute sparse flow.
      memset(flow_stats_, 0);
      // run_kernel1d_functor(compute_flow_stats_kernel<P, flow_stats_t>(pset, flow_stats_, flow_ratio),
      // 			   pset.size(), typename P::architecture());
      run_kernel2d_functor(compute_flow_stats_kernel<P, flow_stats_t>(pset, flow_stats_, flow_ratio),
			   flow_stats_.domain(), typename P::architecture());
      // Fusion with upper flow.
      if (upper_)
	run_kernel2d_functor(flow_fusion_kernel<flow_stats_t, flow_t>
			     (flow_stats_, upper_->flow_stats_, flow_, upper_->flow_),
			     flow_.domain(), typename P::architecture());
      else
	run_kernel2d_functor(flow_fusion_kernel_root<flow_stats_t, flow_t>(flow_stats_, flow_),
			     flow_.domain(), typename P::architecture());

      if (!(frame_cpt_ % filtering_frequency_))
      {
	START_PROF(merge_trajectories);

	run_kernel1d_functor(merge_trajectories_kernel<P>(pset),
			     pset.dense_particles().size(),
			     typename particles_type::architecture());

	END_PROF(merge_trajectories);

	// ****** Filter bad particles.
	START_PROF(filter_spacial_incoherences);

	run_kernel1d_functor(filter_bad_particles_kernel<P>(pset),
			     pset.dense_particles().size(),
			     typename particles_type::architecture());

	END_PROF(filter_spacial_incoherences);
      }

      pset.after_matching();
    }

    template<typename F, typename D, typename P, typename I>
    inline void
    generic_strategy<F, D, P, I>::new_particles(particles_type& pset)
    {
      feature_.bind();
      detector_.new_particles(feature_, pset);
      // cudaUnbindTexture(bc2s_tex_s1);
      // cudaUnbindTexture(bc2s_tex_s2);
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
