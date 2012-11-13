#ifndef CUIMG_MDFL_DETECTOR_HPP_
# define CUIMG_MDFL_DETECTOR_HPP_

# include <cuimg/mt_apply.h>

namespace cuimg
{

  namespace mdfl
  {

    template <typename I>
    inline float compute_contrast(i_short2 p, const I& in, int scale)
    {
      float contrast = 0.f;
      float pv = in(p).x;
      for(int i = 0; i < 8; i++)
	contrast += fabs(pv - in(p + i_int2(c8[i]) * scale).x);
      return contrast;
    }

    template <typename I>
    inline float compute_dfl(i_short2 p, const I& in, int scale)
    {
      float min_diff = 9999999.f;
      float pv = in(p);
      for(int i = 0; i < 8; i++)
      {
	float v1 = in(p + i_int2(circle_r3_h[i]) * scale).x;
	float v2 = in(p + i_int2(circle_r3_h[i+8]) * scale).x;

	float dev = fabs(pv - (v1 + v2) / 2.f);
	if (dev < min_diff)
	  min_diff = dev;
      }

      return min_diff;
    }


  }

  mdfl_1s_detector::mdfl_1s_detector(const obox2d& d)
    : saliency_(d)
  {
  }

  void
  mdfl_1s_detector::update(const host_image2d<i_uchar1>& input)
  {
    mt_apply2d(sizeof(i_float1), input.domain(), [this, &input] (i_int2 p)
	       {
		 saliency_(p) = mdfl::compute_contrast(p, input, 1);
	       }, arch::cpu());
  }

  template <typename F, typename PS>
  void
  mdfl_1s_detector::new_particles(const F& feature, PS& pset, float contrast_th, float dev_th)
  {
    mt_apply2d(sizeof(i_float1), saliency_.domain(),
	       [this, &feature, &pset, contrast_th, dev_th] (i_int2 p)
		{
		  if (pset.has(p)) return;
		  if (saliency_(p) < contrast_th) return;

		  for (int i = 0; i < 8; i++)
		  {
		    i_int2 n(p + i_int2(c8[i]));
		    if (saliency_(p) < saliency_(n) || pset.has(n))
		      return;
		  }

		  // FIXME
		  pset.add(p, feature(p));
		}, arch::cpu());
  }

}

#endif





