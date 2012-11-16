#ifndef CUIMG_MDFL_DETECTOR_HPP_
# define CUIMG_MDFL_DETECTOR_HPP_

# include <cuimg/mt_apply.h>

namespace cuimg
{

  namespace mdfl
  {

    template <typename I>
    inline float compute_saliency(i_short2 p, const I& in, int scale, float contrast_thresh)
    {

      float min_diff = 9999999.f;
      unsigned mean_diff = 0.f;
      float mean_diff_f = 0.f;
      unsigned char pv = in(p).x;
      float max_contrast = 0.f;
      for(int i = 0; i < 8; i++)
      {
        unsigned char v1 = in(p + i_int2(circle_r3_h[i]) * scale).x;
        unsigned char v2 = in(p + i_int2(circle_r3_h[i+8]) * scale).x;

	float contrast = std::max(fabs(pv - v1), fabs(pv - v2));
	if (max_contrast < contrast) max_contrast = contrast;

        unsigned  dev = ::abs(pv - (v1 + v2) / 2);

	mean_diff += dev;
      }

      if (max_contrast >= contrast_thresh)
      {
        min_diff = min_diff / max_contrast;
        mean_diff_f = mean_diff / (8.f * max_contrast);
      }
      else
      {
        min_diff = 0.f;
        mean_diff = 0.f;
      }

      return mean_diff_f;
    }

    template <typename I>
    inline float compute_dfl(i_short2 p, const I& in, int scale)
    {
      float min_diff = 9999999.f;
      float pv = in(p).x;
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

    // template <typename I>
    // inline float compute_saliency(i_short2 p, const I& in, int scale)
    // {
    //   float min_diff = 9999999.f;
    //   float pv = in(p).x;
    //   float max_contrast = 0.f;
    //   for(int i = 0; i < 8; i++)
    //   {
    //     float v1 = in(p + i_int2(circle_r3_h[i]) * scale).x;
    //     float v2 = in(p + i_int2(circle_r3_h[i+8]) * scale).x;

    //     float dev = fabs(pv - (v1 + v2) / 2.f);
    //     if (dev < min_diff)
    //       min_diff = dev;

    // 	float contrast = std::max(fabs(pv - v1), fabs(pv - v2));
    // 	if (max_contrast < contrast) max_contrast = contrast;

    //   }

    //   return min_diff;
    // }
  }

  mdfl_1s_detector::mdfl_1s_detector(const obox2d& d)
    : saliency_(d)
  {
  }

  mdfl_1s_detector&
  mdfl_1s_detector::set_contrast_threshold(float f)
  {
    contrast_th_ = f;
    return *this;
  }

  mdfl_1s_detector&
  mdfl_1s_detector::set_dev_threshold(float f)
  {
    dev_th_ = f;
    return *this;
  }

  void
  mdfl_1s_detector::update(const host_image2d<gl8u>& input)
  {
    START_PROF(mdfl_compute_saliency);
    mt_apply2d(sizeof(i_float1), input.domain() - border(8),
               [this, &input] (i_int2 p)
               {
                 saliency_(p) = mdfl::compute_saliency(p, input, 1, contrast_th_);
               }, arch::cpu());

    END_PROF(mdfl_compute_saliency);

    if (input.nrows() == 480)
      DISPLAY(saliency_);
  }

  template <typename F, typename PS>
  void
  mdfl_1s_detector::new_particles(const F& feature, PS& pset)
  {
    SCOPE_PROF(mdfl_new_particles_detector);
    st_apply2d(sizeof(i_float1), saliency_.domain() - border(8),
               [this, &feature, &pset] (i_int2 p)
               {
                 if (pset.has(p)) return;
                 if (saliency_(p) < dev_th_) return;

                 for (int i = 0; i < 8; i++)
                 {
                   i_int2 n(p + i_int2(c8[i]));
                   if (saliency_(p) < saliency_(n) || pset.has(n))
                     return;
                 }

                 // FIXME Curvature
		 // float r = mdfl::compute_dfl(p, feature.s1(), 1) / (255.f * saliency_(p));
		 // if (r > dev_th)
     #pragma omp critical
		   pset.add(p, feature(p));
               }, arch::cpu());
  }

}

#endif
