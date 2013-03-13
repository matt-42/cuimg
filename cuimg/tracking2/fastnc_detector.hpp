#ifndef CUIMG_FASTNC_DETECTOR_HPP_
# define CUIMG_FASTNC_DETECTOR_HPP_


# include <cuimg/mt_apply.h>
# include <cuimg/run_kernel.h>
# include <cuimg/gpu/cuda.h>
# include <cmath>

namespace cuimg
{

  namespace fastnc
  {

    #ifdef NO_CUDA
    template <typename T>
    inline const T& max(T& a, T& b)
    {
      return std::max(a, b);
    }
    #endif

    template <typename I>
    inline __host__ __device__
    gl8u compute_contrast(i_short2 p, const I& input)
    {
      const int d = 5;
      return ::abs(int(input(p + i_int2(0,d))) - int(input(p + i_int2(0,-d)))) +
	::abs(int(input(p + i_int2(-d,0))) - int(input(p + i_int2(d,0))));
    }

    template <typename I, typename A>
    inline __host__ __device__
    gl8u compute_saliency(i_short2 p, I& input_, int n_, int fast_th_, const A&)
    {

      unsigned max_n = 0;
      unsigned n_bright = 0;
      unsigned n_dark = 0;
      gl8u vp = input_(p);
      unsigned sum_bright = 0;
      unsigned sum_dark = 0;

      for (int i = 0; i < 16; i++)
      {
	gl8u vn = input_(p + i_int2(arch_neighb2d<A>::get(circle_r3_h, circle_r3, i)));
	unsigned char dist = ::abs(int(vn) - int(vp));
	if (dist > fast_th_)
	{
	  if (int(vn) >= int(vp))
	  {
	    n_bright++;
	    sum_bright += dist;
	  }
	  else
	  {
	  n_dark++;
	  sum_dark += dist;
	  }
	}
      }
      max_n = std::max(n_dark, n_bright);

      if (max_n < n_)
	return 0;
      else
	return max(sum_bright, sum_dark);

    }
  }

  template <typename A>
  fastnc_detector<A>::fastnc_detector(const obox2d& d)
    : n_(9),
      saliency_(d),
      contrast_(d),
      new_points_(d)
  {
  }

  template <typename A>
  fastnc_detector<A>&
  fastnc_detector<A>::set_fast_threshold(float f)
  {
    fast_th_ = f;
    return *this;
  }

  template <typename A>
  fastnc_detector<A>&
  fastnc_detector<A>::set_n(unsigned n)
  {
    n_ = n;
    return *this;
  }

  template <typename A>
  template <typename F, typename PS>
  void
  fastnc_detector<A>::new_particles(F& feature, PS& pset)
  {
    new_particles(feature, pset, A());
  }

#ifndef NO_CPP0X
  template <typename A>
  template <typename J>
  void
  fastnc_detector<A>::update(const image2d_gl8u& input, const J& mask)
  {
    input_ = input;

    mt_apply2d(sizeof(i_float1), input.domain() - border(8),
	       [this, &input] (i_int2 p)
	       {
		 this->saliency_(p) = fastnc::compute_saliency(p, input, this->n_, this->fast_th_, cpu());
	       }, cpu());

    // contrast
    mt_apply2d(sizeof(i_float1), input.domain() - border(5),
	       [this, &input] (i_int2 p)
	       {
		 contrast_(p) = fastnc::compute_contrast(p, input);
	       }, cpu());

  }


  template <typename A>
  template <typename F, typename PS>
  void
  fastnc_detector<A>::new_particles(F& feature, PS& pset, const cpu&)
  {
    SCOPE_PROF(fastnc_new_particles_detector);
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

#ifndef NO_CUDA
#endif

}

#endif
