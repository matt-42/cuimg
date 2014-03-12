#ifndef CUIMG_FAST_DETECTOR_HPP_
# define CUIMG_FAST_DETECTOR_HPP_


# include <cuimg/mt_apply.h>
# include <cuimg/run_kernel.h>
# include <cuimg/gpu/cuda.h>
# include <cuimg/neighb2d.h>
# include <cuimg/border.h>
# include <cmath>

namespace cuimg
{

  namespace fast
  {

#ifdef NO_CUDA
    template <typename T>
    inline const T& max(T& a, T& b)
    {
      return std::max(a, b);
    }
#endif

    template <typename I, typename A>
    inline __host__ __device__
    gl8u compute_saliency(i_short2 p, I& input_, int n_, int fast_th_, const A&)
    {
      unsigned max_n = 0;
      unsigned n = 0;
      int status = 2;
      gl8u vp = input_(p);
      unsigned sum_bright = 0;
      unsigned sum_dark = 0;

      // if (n_ == 15)
      // {
      // 	int to_test[3] = {0, 4, 12, 0};
      // 	for (int i = 0; i < 3; i++)
      // 	{
      // 		gl8u vn = input_(p + i_int2(arch_neighb2d<A>::get(circle_r3_h, circle_r3, to_test[i])));
      // 		int sign = int(vn) > int(vp);
      // 		unsigned char dist = ::abs(int(vn) - int(vp));

      // 		if (dist > fast_th_)
      // 			if (sign == status) n++;
      // 			else
      // 			{
      // 				if (n > max_n)
      // 					max_n = n;
      // 				status = sign;
      // 				n = 1;
      // 			}
      // 		else
      // 		{
      // 			if (n > max_n)
      // 				max_n = n;
      // 			status = 2;
      // 		}
      // 	}
      // 	if (n < 3) return 0;
      // }

      if (n_ >= 9)
      {
      	int to_test[6] = {0, 4, 8, 12, 0, 4};
      	int equals = 0;
      	for (int i = 0; i < 6; i++)
      	{
      	  gl8u vn = input_(p + i_int2(arch_neighb2d<A>::get(circle_r3_h, circle_r3, to_test[i])));
      	  int sign = int(vn) > int(vp);
      	  unsigned char dist = ::abs(int(vn) - int(vp));

      	  if (dist > fast_th_)
      	    if (sign == status) { n++; }
      	    else
      	    {
      	      if (n > max_n)
      		max_n = n;
      	      status = sign;
      	      n = 1;
      	    }
      	  else
      	  {
      	    if (n > max_n)
      	      max_n = n;
      	    status = 2;
      	    equals++;
      	    if (i < 4 && n_ >= 12 && equals >= 2) return 0;
      	  }
      	}
      	if (n > max_n)
      	  max_n = n;
      	if (n_ >= 12 && max_n < 3) return 0;
      	else if (n_ >= 8 && max_n < 2) return 0;
      }

      n = 0;
      status = 2;
      max_n = 0;
      for (int i = 0; i < 16; i++)
      {
	gl8u vn = input_(p + i_int2(arch_neighb2d<A>::get(circle_r3_h, circle_r3, i)));
	int sign = int(vn) > int(vp);
	unsigned char dist = ::abs(int(vn) - int(vp));
	if (dist > fast_th_)
	{
	  if (sign == status) n++;
	  else
	  {
	    if (n > max_n)
	      max_n = n;
	    status = sign;
	    n = 1;
	  }

	  if (vn < vp)
	    sum_dark += dist;
	  else
	    sum_bright += dist;
	}
	else
	{
	  if (n > max_n)
	    max_n = n;
	  status = 2;
	}
      }

      if (n != 16 && status != 2)
      {
	int i = 0;
	while (true)
	{
	  gl8u vn = input_(p + i_int2(arch_neighb2d<A>::get(circle_r3_h, circle_r3, i)));
	  int sign = int(vn) > int(vp);
	  unsigned char dist = ::abs(int(vn) - int(vp));

	  if (dist <= fast_th_ || sign != status) break;

	  n++;
	  i++;
	  assert(i < 16);
	}

      }

      if (n > max_n)
	max_n = n;

      if (max_n < n_)
	return 0;
      else
	return max(sum_bright, sum_dark);
	// return sum_bright;
    }
  }

  template <typename A>
  fast_detector<A>::fast_detector(const obox2d& d)
    : n_(9),
      box_size_(3),
      saliency_(d, 1),
      new_points_(d),
      input_s2_(d, 3)
  {
  }

  template <typename A>
  fast_detector<A>&
  fast_detector<A>::set_fast_threshold(float f)
  {
    fast_th_ = f;
    return *this;
  }


  template <typename A>
  fast_detector<A>&
  fast_detector<A>::set_box_size(unsigned b)
  {
    box_size_ = b;
    return *this;
  }

  template <typename A>
  fast_detector<A>&
  fast_detector<A>::set_n(unsigned n)
  {
    n_ = n;
    return *this;
  }

  template <typename A>
  template <typename F, typename PS>
  void
  fast_detector<A>::new_particles(F& feature, PS& pset)
  {
    new_particles(feature, pset, A());
  }

#ifndef NO_CPP0X
  // template <typename A>
  // void
  // fast_detector<A>::update(const host_image2d<gl8u>& input)
  // {
  //   input_ = input;

  //   mt_apply2d(sizeof(i_float1), input.domain() - border(8),
  // 	       [this, &input] (i_int2 p)
  // 	       {
  // 		 this->saliency_(p) = fast::compute_saliency(p, input, this->n_, this->fast_th_, cpu());
  // 	       }, cpu());
  // }

  template <typename A>
  template <typename F, typename PS>
  void
  fast_detector<A>::new_particles(F& feature, PS& pset, const cpu&)
  {
    SCOPE_PROF(fast_new_particles_detector);
    memset(new_points_, 0);
    typename PS::kernel_type pset_ = pset;
    mt_apply2d(sizeof(i_float1), saliency_.domain() / box_size_,
               [this, &feature, &pset_] (i_int2 p)
               {
		 p = (p) * box_size_ + i_int2(1,1);
		 float vmax = fast::compute_saliency(p, input_, n_, fast_th_, A());
		 i_int2 max_p = p;

                 for (int r = 0; r < box_size_; r++)
                 for (int c = 0; c < box_size_; c++)
                 {
                   i_int2 n = p + i_int2(r, c);
                   if (pset_.has(n)) return;
                 }

                 for (int r = 0; r < box_size_; r++)
                 for (int c = 0; c < box_size_; c++)
                 {
                   i_int2 n = p + i_int2(r, c);
                   float s = fast::compute_saliency(n, input_, n_, fast_th_, A());
		   if (vmax < s) { vmax = s; max_p = n; }
                 }
                 if (vmax < fast_th_) return;

                 new_points_(max_p) = max_p;
               }, cpu());

    st_apply2d(sizeof(i_float1), saliency_.domain() - border(0),
               [this, &feature, &pset] (i_int2 p)
               {
                 if (new_points_(p) != i_short2(0,0)) pset.add(p, feature(p));
               }, cpu());

  }


#endif

  template <typename I, typename J>
  struct compute_saliency_kernel
  {
    typedef typename I::architecture architecture;
    const typename I::kernel_type input_;
    const typename J::kernel_type mask_;
    typename I::kernel_type saliency_;
    int n_;
    float fast_th_;

    compute_saliency_kernel(const I input, const J mask, I saliency, int n, float fast_th)
      : input_(input),
	mask_(mask),
	saliency_(saliency),
	n_(n),
	fast_th_(fast_th)
    {
      assert(input.border() >= 3);
    }

    __host__ __device__ inline
    void operator()(i_int2 p)
    {
      if (mask_(p))
	saliency_(p) = fast::compute_saliency(p, input_, n_, fast_th_, architecture());
    }

  };

  template <typename A>
  template <typename J>
  void
  fast_detector<A>::update(const image2d_gl8u& input, const J& mask)
  {
    SCOPE_PROF(fast_compute_saliency);
    input_ = input;

    // cv::Mat opencv_s2(input_s2_);
    // cv::GaussianBlur(cv::Mat(input), opencv_s2, cv::Size(11, 11), 1, 1, cv::BORDER_REPLICATE);
    //copy(input, input_s2_);
    //fill_border_clamp(input_s2_);

    // memset(saliency_, 0);
    // run_kernel2d_functor(compute_saliency_kernel<image2d_gl8u, J>(input, mask, saliency_, n_, fast_th_),
    //     		 input.domain(), A());

  }

#ifndef NO_CUDA

  template <typename PS, typename I, typename J>
  __global__
  void select_particles(PS pset, const kernel_image2d<I> saliency, kernel_image2d<J> new_points, obox2d domain)
  {
    i_int2 p = thread_pos2d();

    p = (p) * 3 + i_int2(1,1);

    if (!saliency.has(p)) return;

    float vmin = saliency(p);
    i_int2 min_p = p;
    for_all_in_static_neighb2d(p, n, c8)
      if (vmin < saliency(n)) { vmin = saliency(n); min_p = n; }

    if (pset.has(min_p)) return;
    if (saliency(min_p) == 0) return;

    new_points(min_p) = min_p;
  }


#ifdef NVCC
  template <typename A>
  template <typename F, typename PS>
  void
  fast_detector<A>::new_particles(F& feature, PS& pset, const cuda_gpu&)
  {
    SCOPE_PROF(fast_new_particles_detector);
    memset(new_points_, 0);


    select_particles<<<A::dimgrid2d(saliency_.domain()/3), A::dimblock2d()>>>
      (typename PS::kernel_type(pset), mki(saliency_), mki(new_points_),
       new_points_.domain());

    pset.append_new_points(new_points_, feature);
  }
#endif

#endif

}

#endif
