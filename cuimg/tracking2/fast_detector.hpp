#ifndef CUIMG_FAST_DETECTOR_HPP_
# define CUIMG_FAST_DETECTOR_HPP_

# include <cuimg/mt_apply.h>
# include <cuimg/gpu/cuda.h>

namespace cuimg
{

  namespace fast
  {

    template <typename I>
    inline __host__ __device__
    gl8u compute_contrast(i_short2 p, const I& input, I& contrast_)
    {
      const int d = 5;
      contrast_(p) = ::abs(int(input(p + i_int2(0,d))) - int(input(p + i_int2(0,-d)))) +
	::abs(int(input(p + i_int2(-d,0))) - int(input(p + i_int2(d,0))));
    }

    template <typename I>
    inline __host__ __device__
    gl8u compute_saliency(i_short2 p, const I& input_, int n_, int fast_th_)
    {
      unsigned max_n = 0;
      unsigned n = 0;
      int status = 2;
      gl8u vp = input_(p);
      unsigned sum_bright = 0;
      unsigned sum_dark = 0;

      for (int i = 0; i < 16; i++)
      {
	gl8u vn = input_(p + i_int2(circle_r3_h[i]));
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
	  gl8u vn = input_(p + i_int2(circle_r3_h[i]));
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
	return std::max(sum_bright, sum_dark);
    }
  }

  template <typename A>
  fast_detector<A>::fast_detector(const obox2d& d)
    : n_(9),
      saliency_(d),
      contrast_(d),
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
  fast_detector<A>::new_particles(const F& feature, PS& pset)
  {
    new_particles(feature, pset, A());
  }

#ifndef NO_CPP0X
  template <typename A>
  void
  fast_detector<A>::update(const host_image2d<gl8u>& input)
  {
    input_ = input;

    mt_apply2d(sizeof(i_float1), input.domain() - border(8),
	       [this, &input] (i_int2 p)
	       {
		 this->saliency_(p) = fast::compute_saliency(p, input, this->n_, this->fast_th_);
	       }, cpu());

    // contrast
    mt_apply2d(sizeof(i_float1), input.domain() - border(5),
	       [this, &input] (i_int2 p)
	       {
		 contrast_(p) = fast::compute_contrast(p, input, contrast_);
	       }, cpu());

  }


  template <typename A>
  template <typename F, typename PS>
  void
  fast_detector<A>::new_particles(const F& feature, PS& pset, const cpu&)
  {
    SCOPE_PROF(fast_new_particles_detector);
    memset(new_points_, 0);
    mt_apply2d(sizeof(i_float1), saliency_.domain() - border(8),
               [this, &feature, &pset] (i_int2 p)
               {
                 if (pset.has(p)) return;
                 if (saliency_(p) == 0) return;

                 for (int i = 0; i < 8; i++)
                 {
                   i_int2 n(p + i_int2(c8[i]));
                   if (saliency_(p) < saliency_(n) || pset.has(n))
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

#ifndef NO_CUDA

  template <typename PS, typename I, typename J>
  __global__
  void select_particles(PS pset, const I saliency, J new_points)
  {
    i_int2 p = thread_pos2d();

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
  fast_detector<A>::new_particles(const F& feature, PS& pset, const cuda_gpu&)
  {
    SCOPE_PROF(fast_new_particles_detector);
    memset(new_points_, 0);

    //dim3 dimblock = A::dimblock();
    //dim3 dimgrid = ;
    select_particles<<<A::dimgrid(saliency_.domain()), A::dimblock()>>>
      (pset, saliency_, new_points_);

    pset.append_new_points(new_points_, feature);
  }
#endif

#endif

}

#endif
