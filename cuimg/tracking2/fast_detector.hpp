#ifndef CUIMG_FAST_DETECTOR_HPP_
# define CUIMG_FAST_DETECTOR_HPP_


# include <cuimg/mt_apply.h>
# include <cuimg/run_kernel.h>
# include <cuimg/gpu/cuda.h>
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
	  status = 2;
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
    }
  }

  template <typename A>
  fast_detector<A>::fast_detector(const obox2d& d)
    : n_(9),
      saliency_(d),
      new_points_(d)
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
    mt_apply2d(sizeof(i_float1), saliency_.domain() - border(8),
               [this, &feature, &pset_] (i_int2 p)
               {
                 if (pset_.has(p)) return;
                 if (saliency_(p) == 0) return;

                 for (int i = 0; i < 8; i++)
                 {
                   i_int2 n(p + i_int2(c8_h[i]));
                   if (saliency_(p) < saliency_(n) || pset_.has(n))
                     return;
                 }

                 new_points_(p) = p;
               }, cpu());

    st_apply2d(sizeof(i_float1), saliency_.domain() - border(8),
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

    compute_saliency_kernel(const I input, const I mask, I saliency, int n, float fast_th)
      : input_(input),
	mask_(mask),
	saliency_(saliency),
	n_(n),
	fast_th_(fast_th)
    {
    }

    __host__ __device__ inline
    void operator()(i_int2 p)
    {
      if (!mask_(p))
	saliency_(p) = fast::compute_saliency(p, input_, n_, fast_th_, architecture());
    }

  };

  template <typename A>
  template <typename J>
  void
  fast_detector<A>::update(const image2d_gl8u& input, const J& mask)
  {
    input_ = input;
    memset(saliency_, 0);
    run_kernel2d_functor(compute_saliency_kernel<image2d_gl8u, J>(input, mask, saliency_, n_, fast_th_),
			 input.domain() - border(4), A());
  }

#ifndef NO_CUDA

  template <typename PS, typename I, typename J>
  __global__
  void select_particles(PS pset, const kernel_image2d<I> saliency, kernel_image2d<J> new_points, box2d domain)
  {
    i_int2 p = thread_pos2d();

    if (!domain.has(p)) return;

    if (pset.has(p)) return;
    if (saliency(p) == 0) return;


    for (int i = 0; i < 8; i++)
    {
      i_int2 n(p + i_int2(c8[i]));
      if (saliency(p) < saliency(n) || pset.has(n))
	return;
    }

    new_points(p) = p;
  }


#ifdef NVCC
  template <typename A>
  template <typename F, typename PS>
  void
  fast_detector<A>::new_particles(F& feature, PS& pset, const cuda_gpu&)
  {
    SCOPE_PROF(fast_new_particles_detector);
    memset(new_points_, 0);


    select_particles<<<A::dimgrid2d(saliency_.domain()), A::dimblock2d()>>>
      (typename PS::kernel_type(pset), mki(saliency_), mki(new_points_),
       new_points_.domain() - border(6));

    pset.append_new_points(new_points_, feature);
  }
#endif

#endif

}

#endif
