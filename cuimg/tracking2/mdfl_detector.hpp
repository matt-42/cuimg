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

	unsigned dev = ::abs(pv - (v1 + v2) / 2);
	// unsigned dev = ::abs(pv - v1) + ::abs(pv - v2);

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
    inline float compute_saliency(i_short2 p, const I& in, int scale, int a, int b, int c, int d)
    {
      unsigned char v1 = in(p + i_int2(circle_r3_h[a]) * scale).x;
      unsigned char v2 = in(p + i_int2(circle_r3_h[b]) * scale).x;
      unsigned char v3 = in(p + i_int2(circle_r3_h[c]) * scale).x;
      unsigned char v4 = in(p + i_int2(circle_r3_h[d]) * scale).x;

      return std::min(fabs(v1 - v2), fabs(v3 - v4));
    }

    template <typename I>
    inline float compute_saliency_test(i_short2 p, const I& in, int scale, float contrast_thresh)
    {

      float min_diff = 9999999.f;
      unsigned mean_diff = 0.f;
      float mean_diff_f = 0.f;
      unsigned char pv = in(p).x;
      float max_contrast = 0.f;

      for(int i = 0; i < 16; i++)
      {
	float contrast = fabs(pv - in(p + i_int2(circle_r3_h[i]) * scale).x);
	if (max_contrast < contrast) max_contrast = contrast;
      }

      for(int i = 2; i < 6; i++)
      {
        unsigned char v1 = in(p + i_int2(circle_r3_h[i+2]) * scale).x;
        unsigned char v2 = in(p + i_int2(circle_r3_h[i+8-2]) * scale).x;

        unsigned char v3 = in(p + i_int2(circle_r3_h[i-2]) * scale).x;
        unsigned char v4 = in(p + i_int2(circle_r3_h[i+8+2]) * scale).x;

        unsigned dev = compute_saliency(p, in, scale, i+2, i+8-2, i-2, i+8+2);
	mean_diff += dev;
	if (dev < min_diff) min_diff = dev;
      }

      unsigned dev;
      dev = compute_saliency(p, in, scale, 14, 10, 2, 6); // 0
      mean_diff += dev; if (dev < min_diff) min_diff = dev;
      dev = compute_saliency(p, in, scale, 15, 11, 3, 7); // 1
      mean_diff += dev; if (dev < min_diff) min_diff = dev;
      dev = compute_saliency(p, in, scale,  8,  12, 4, 0); // 6
      mean_diff += dev; if (dev < min_diff) min_diff = dev;
      dev= compute_saliency(p, in, scale,  9,  13, 5, 1); // 7
      mean_diff += dev; if (dev < min_diff) min_diff = dev;

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

      return min_diff;
      // return mean_diff_f / 8.f;
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
    : saliency_(d),
      new_points_(d)
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

    // if (input.nrows() == 480)
    //   DISPLAY(saliency_);
  }

  template <typename F, typename PS>
  void
  mdfl_1s_detector::new_particles(const F& feature, PS& pset)
  {
    SCOPE_PROF(mdfl_new_particles_detector);
    memset(new_points_, 0);
    mt_apply2d(sizeof(i_float1), saliency_.domain() - border(8),
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
                 new_points_(p) = 1;
               }, arch::cpu());

    st_apply2d(sizeof(char), saliency_.domain() - border(8),
               [this, &feature, &pset] (i_int2 p)
               {
                 if (new_points_(p)) pset.add(p, feature(p));
               }, arch::cpu());

  }

}

#endif
