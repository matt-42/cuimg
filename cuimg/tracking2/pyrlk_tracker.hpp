#ifndef CUIMG_PYRLK_TRACKING_STRATEGIES_HPP_
# define CUIMG_PYRLK_TRACKING_STRATEGIES_HPP_

#include <Eigen/Dense>
#include <cuimg/tracking2/tracking_strategies.hpp>

namespace cuimg
{
  namespace tracking_strategies
  {

    template <typename F, typename GD>
    inline __host__ __device__
    std::pair<i_short2, float> iterative_lk_match(i_short2 p, i_short2 prediction_, F A, F B, GD Ag)
    {
      // Image difference.

      // Gradient matrix

      Eigen::Matrix2d G = Eigen::Matrix2d::Zero();
      int cpt = 0;
      for(int i = 0; i < 16; i ++)
      {
	i_int2 n = p + i_int2(circle_r3[i]);

	if (A.has(n))
	{
	  Eigen::Matrix2d m;
	  double gx = Ag(n)[0] / 2.;
	  double gy = Ag(n)[1] / 2.;
	  m <<
	    gx * gx, gx * gy,
	    gx * gy, gy * gy;
	  G += m;
	  cpt++;
	}
      }

      G /= cpt;

      //if (fabs(G.determinant()) < 0.0001)
      auto ev = G.eigenvalues();
      float min_ev = 999999;
      for (int i = 0; i < ev.size(); i++)
	if (fabs(ev[i].real()) < min_ev) min_ev = fabs(ev[i].real());

      if (min_ev < 0.0001)
      {
      	return std::pair<i_short2, float>(i_short2(-1,-1), 1000);
      }

      Eigen::Matrix2d G1 = G.inverse();

      i_int2 v = prediction_;
      Eigen::Vector2d nk = Eigen::Vector2d::Ones();

      i_int2 gs[16];
      i_uchar1 as[16];
      for(int i = 0; i < 16; i ++)
      {
	i_int2 n = p + i_int2(circle_r3[i]);
	if (Ag.has(n))
	{
	  gs[i] = Ag(n);
	  as[i] = A(n);
	}
      }

      for (int k = 0; k < 10 && nk.norm() >= 1; k++)
      {
	Eigen::Vector2d bk = Eigen::Vector2d::Zero();
	// Temporal difference.
	cpt = 0;
	for(int i = 0; i < 16; i ++)
	{
	  i_int2 n1 = p + i_int2(circle_r3[i]);
	  i_int2 n2 = v + i_int2(circle_r3[i]);
	  if (A.has(n1) and B.has(n2))
	  {
	    auto& g = gs[i];
	    double dt = (as[i] - B(n2)).x;
	    bk += Eigen::Vector2d(g[0] * dt, g[1] * dt);
	    cpt++;
	  }
	}
	bk /= cpt;

	nk = G1 * bk;
	v += i_int2(nk[0], nk[1]);
      }

      float err = 0;
      for(int i = 0; i < 16; i ++)
      {
	i_int2 n1 = p + i_int2(circle_r3[i]);
	i_int2 n2 = v + i_int2(circle_r3[i]);
	if (A.has(n1) and B.has(n2))
	{
	  err += fabs((as[i] - B(n2)).x);
	  cpt++;
	}
      }

      //std::cout << v << std::endl;
      return std::pair<i_short2, float>(v, err / cpt);
    }

    template <typename F, typename GD>
    inline __host__ __device__
    std::pair<i_short2, float> iterative_lk_match2(i_short2 p, i_short2 prediction_, F A, F B, GD Ag)
    {
      // Image difference.

      // Gradient matrix

      Eigen::Matrix2f G = Eigen::Matrix2f::Zero();
      int cpt = 0;
      for(int i = 0; i < 8; i ++)
      {
	i_int2 n = p + i_int2(circle_r3[i*2]);

	if (A.has(n))
	{
	  Eigen::Matrix2f m;
	  float gx = Ag(n)[0] / 2;
	  float gy = Ag(n)[1] / 2;
	  m <<
	    gx * gx, gx * gy,
	    gx * gy, gy * gy;
	  G += m;
	  cpt++;
	}
      }

      G /= cpt;

      //if (fabs(G.determinant()) < 0.0001)
      auto ev = G.eigenvalues();
      float min_ev = 999999;
      for (int i = 0; i < ev.size(); i++)
	if (fabs(ev[i].real()) < min_ev) min_ev = fabs(ev[i].real());

      std::cout << min_ev<< std::endl;
      if (min_ev < 0.001)
      {
      	return std::pair<i_short2, float>(i_short2(-1,-1), 1000);
      }

      Eigen::Matrix2f G1 = G.inverse();

      i_int2 v = prediction_;
      Eigen::Vector2f nk = Eigen::Vector2f::Ones();

      for (int k = 0; k < 10 && nk.norm() >= 1; k++)
      {
	Eigen::Vector2f bk = Eigen::Vector2f::Zero();
	// Temporal difference.
	cpt = 0;
	for(int i = 0; i < 8; i ++)
	{
	  i_int2 n1 = p + i_int2(circle_r3[i*2]);
	  i_int2 n2 = v + i_int2(circle_r3[i*2]);
	  if (A.has(n1) and B.has(n2))
	  {
	    auto g = Ag(n1);
	    float dt = (A(n1) - B(n2)).x;
	    bk += Eigen::Vector2f(g[0] * dt, g[1] * dt);
	    cpt++;
	  }
	}
	bk /= cpt;

	nk = G1 * bk;
	v += i_int2(nk[0], nk[1]);
      }

      //std::cout << v << std::endl;
      return std::pair<i_short2, float>(v, 0.f);
    }


    template <typename F, typename GD>
    inline __host__ __device__
    std::pair<i_short2, float> iterative_lk_match3(i_short2 p, i_short2 prediction_, F A, F B, GD Ag)
    {
      // Image difference.

      // Gradient matrix

      Eigen::Matrix2f G = Eigen::Matrix2f::Zero();
      int cpt = 0;
      for(int i = 0; i < 8; i ++)
      {
	i_int2 n = p + i_int2(c8_h[i]);

	if (A.has(n))
	{
	  Eigen::Matrix2f m;
	  float gx = Ag(n)[0] / 2;
	  float gy = Ag(n)[1] / 2;
	  m <<
	    gx * gx, gx * gy,
	    gx * gy, gy * gy;
	  G += m;
	  cpt++;
	}
      }

      G /= cpt;

      //if (fabs(G.determinant()) < 0.0001)
      auto ev = G.eigenvalues();
      float min_ev = 999999;
      for (int i = 0; i < ev.size(); i++)
	if (fabs(ev[i].real()) < min_ev) min_ev = fabs(ev[i].real());

      if (min_ev < 0.0001)
      {
      	return std::pair<i_short2, float>(i_short2(-1,-1), 1000);
      }

      Eigen::Matrix2f G1 = G.inverse();

      i_int2 v = prediction_;
      Eigen::Vector2f nk = Eigen::Vector2f::Ones();

      for (int k = 0; k < 10 && nk.norm() >= 1; k++)
      {
	Eigen::Vector2f bk = Eigen::Vector2f::Zero();
	// Temporal difference.
	cpt = 0;
	for(int i = 0; i < 8; i ++)
	{
	  i_int2 n1 = p + i_int2(c8_h[i]);
	  i_int2 n2 = v + i_int2(c8_h[i]);
	  if (A.has(n1) and B.has(n2))
	  {
	    auto g = Ag(n1);
	    float dt = (A(n1) - B(n2)).x;
	    bk += Eigen::Vector2f(g[0] * dt, g[1] * dt);
	    cpt++;
	  }
	}
	bk /= cpt;

	nk = G1 * bk;
	v += i_int2(nk[0], nk[1]);
      }

      //std::cout << v << std::endl;
      return std::pair<i_short2, float>(v, 0.f);
    }

    template<typename I, typename J, typename C, typename GI, typename P>
    struct pyrlk_match_particles_kernel
    {
    private:
      typename P::kernel_type pset;
      typename I::kernel_type input;
      typename I::kernel_type prev_input;
      typename GI::kernel_type gradient;
      typename J::kernel_type upper_flow;
      typename C::kernel_type contrast;
      int flow_ratio;
      int k;

    public:
      pyrlk_match_particles_kernel(P& pset_,
				   I& prev_input_,
				   I& input_,
				   GI& gradient_,
				   const J& upper_flow_,
				   C& contrast_,
				   int flow_ratio_, int k_)
	: pset(pset_),
	  prev_input(prev_input_),
	  input(input_),
	  gradient(gradient_),
	  upper_flow(upper_flow_),
	  contrast(contrast_),
	  flow_ratio(flow_ratio_),
	  k(k_)
      {
      }

      inline __host__ __device__ void
      operator()(int i)
      {
	assert(i >= 0 && i < pset.size());
	particle& part = pset.dense_particles()[i];
	assert(pset.domain().has(part.pos));
	box2d domain = pset.domain() - border(0);
	assert(domain.has(part.pos));
	if (part.age > 0)
	{
	  // Prediction.
	  i_short2 pred;
	  pred = multiscale_prediction(part, upper_flow, flow_ratio);

	  // Matching.
	  if (domain.has(pred))
	  {
	    float distance;
	    std::pair<i_short2, float> match_res = iterative_lk_match(part.pos, pred,
								      prev_input,
								      input,
								      gradient);
	    i_short2 match = match_res.first;
	    distance = match_res.second;
	    unsigned cpt = 0;
	    if (domain.has(match) and distance < k)
	    {
	      // if (contrast(match) < 3.f)
	      // 	pset.remove(i);
	      // else
	      {
		pset.move(i, match, 0);
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

    template <typename D>
    pyrlk_cpu<D>::pyrlk_cpu(const obox2d& d)
      : prev_input_(d),
	tmp_(d),
	flow_ratio(2),
	detector_(d),
	upper_(0),
	frame_cpt_(0),
	gradient_(d),
	contrast_(d),
	mask_(d, 4),
	detector_frequency_(1),
	filtering_frequency_(1),
	k_(300),
	with_merge_(true)
    {
      new_points_map_ = gl8u_image2d(domain_div_up(d, 2*flow_ratio));
      flow_stats_ = flow_stats_t(domain_div_up(d, flow_ratio));
      flow_ = flow_t(domain_div_up(d, flow_ratio));
      multiscale_count_ = uint_image2d(domain_div_up(d, flow_ratio));
    }

    template<typename D>
    void
    pyrlk_cpu<D>::set_with_merge(bool b)
    {
      with_merge_ = b;
    }

    template<typename D>
    pyrlk_cpu<D>&
    pyrlk_cpu<D>::set_detector_frequency(unsigned nframe)
    {
      detector_frequency_ = nframe;
      if (upper_)
	upper_->set_detector_frequency(nframe);
      return *this;
    }


    template<typename D>
    pyrlk_cpu<D>&
    pyrlk_cpu<D>::set_filtering_frequency(unsigned nframe)
    {
      filtering_frequency_ = nframe;
      if (upper_)
	upper_->set_filtering_frequency(nframe);
      return *this;
    }

    template<typename D>
    int
    pyrlk_cpu<D>::border_needed() const
    {
      return detector_.border_needed();
    }


    template<typename D>
    void
    pyrlk_cpu<D>::update(const gl8u_image2d& in,
			 PC& pset)
    {
      input_ = in;
      cv::Mat opencv_s1(tmp_);
      cv::GaussianBlur(cv::Mat(in), opencv_s1, cv::Size(3, 3), 1, 1, cv::BORDER_REPLICATE);
      copy(tmp_, input_);

      if (frame_cpt_ < 1)
      {
	copy(in, prev_input_);
	frame_cpt_++;
	return;
      }

      pset.set_flow(flow_);
      match_particles(pset);

      if (!(frame_cpt_ % detector_frequency_))
      {
	create_detector_mask(pset);
	detector_.update(in, mask_);
	new_particles(pset);
      }

      if (!(frame_cpt_ % filtering_frequency_))
      {
	filter(pset);

	START_PROF(merge_trajectories);
	if (with_merge_)
	  run_kernel1d_functor(merge_trajectories_kernel<PC>(pset),
			       pset.dense_particles().size(),
			       typename PC::architecture());
	END_PROF(merge_trajectories);
      }

      pset.tick();
      frame_cpt_++;
      copy(in, prev_input_);
    }



    template<typename D>
    void
    pyrlk_cpu<D>::create_detector_mask(PC& pset)
    {
      memset(mask_, 255);
      run_kernel1d_functor(particle_mask_kernel<PC, gl8u_image2d>
			   (pset, mask_),
			   pset.size(), typename PC::architecture());

    }


    template<typename D>
    void
    pyrlk_cpu<D>::match_particles(PC& pset)
    {
      pset.before_matching();

      // Matching
      START_PROF(matcher);

      memset(gradient_, 0);
      for (i_int2 p : input_.domain() - border(1))
      {
	gradient_(p)[0] = int(prev_input_(p + i_int2(1,0)).x) - int(prev_input_(p + i_int2(-1,0)).x);
	gradient_(p)[1] = int(prev_input_(p + i_int2(0,1)).x) - int(prev_input_(p + i_int2(0,-1)).x);
      }

      flow_t uf = upper_ ? upper_->flow_ : flow_t();
      pyrlk_match_particles_kernel<gl8u_image2d, flow_t, uint_image2d, gradient_t, PC> func
	(pset, prev_input_, input_, gradient_, uf, contrast_, flow_ratio, k_);

      run_kernel1d_functor(func,
			   pset.dense_particles().size(),
			   typename PC::architecture());

      END_PROF(matcher);

      // Compute sparse flow.
      memset(flow_stats_, 0);
      fill(flow_, NO_FLOW);
      memset(multiscale_count_, 0);
      run_kernel2d_functor(compute_flow_stats_kernel<PC, flow_stats_t>(pset, flow_stats_, flow_ratio),
			   flow_stats_.domain(), typename PC::architecture());

      // Fusion with upper flow.
      if (upper_)
	run_kernel2d_functor(flow_fusion_kernel<flow_stats_t, flow_t, uint_image2d>
			     (flow_stats_, upper_->flow_stats_, flow_, upper_->flow_, multiscale_count_),
			     flow_.domain(), typename PC::architecture());
      else
      {
	run_kernel2d_functor(flow_fusion_kernel_root<flow_stats_t, flow_t, uint_image2d>
			     (flow_stats_, flow_, multiscale_count_),
			     flow_.domain(), typename PC::architecture());
      }

      pset.after_matching();
    }



    template<typename D>
    inline void
    pyrlk_cpu<D>::filter(PC& pset)
    {
      if (!(frame_cpt_ % filtering_frequency_))
      {
	// ****** Filter bad particles.
	START_PROF(filter_spacial_incoherences);

	run_kernel1d_functor(filter_bad_particles_kernel<PC, flow_t, uint_image2d>
			     (pset, flow_, multiscale_count_, flow_ratio),
			     pset.dense_particles().size(),
			     typename PC::architecture());

	END_PROF(filter_spacial_incoherences);
      }
    }

    template<typename D>
    inline void
    pyrlk_cpu<D>::new_particles(PC& pset)
    {
      dummy_feature f;
      detector_.new_particles(f, pset);
      pset.after_new_particles();
    }


    template<typename D>
    void
    pyrlk_cpu<D>::set_upper(self* u)
    {
      u->lower_ = this;
      upper_ = u;
    }

    template<typename D>
    pyrlk_cpu<D>*
    pyrlk_cpu<D>::upper()
    {
      return upper_;
    }

    template<typename D>
    void
    pyrlk_cpu<D>::clear()
    {
      frame_cpt_ = 0;
    }

    template<typename D>
    i_int2
    pyrlk_cpu<D>::get_flow_at(const i_int2& p)
    {
      if (flow_(p / flow_ratio).first)
	return flow_(p / flow_ratio).second;
      else
      {
	particle part;
	part.age = 1;
	part.pos = p;
	flow_t uf = upper_ ? upper_->flow_ : flow_t();
	i_short2 pred = multiscale_prediction(part, uf, flow_ratio);
	if ((input_.domain() - border(7)).has(pred))
	{
	  float distance;
	  std::pair<i_short2, float> match_res = iterative_lk_match(part.pos, pred,
								    prev_input_,
								    input_,
								    gradient_);
	  i_short2 match = match_res.first;
	  if (input_.domain().has(match))
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

