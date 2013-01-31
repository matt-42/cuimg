#ifndef CUIMG_MDFL_DETECTOR_HPP_
# define CUIMG_MDFL_DETECTOR_HPP_

# include <cuimg/mt_apply.h>
# include <cuimg/memset.h>

namespace cuimg
{

  namespace mdfl
  {

    template <typename I>
    inline std::pair<int, int> compute_saliency(i_short2 p, const I& in, int scale, float contrast_thresh)
    {

      int min_diff = 999999;
      int mean_diff = 0.f;
      int pv = in(p).x;
      int max_contrast = 0;
      for(int i = 0; i < 8; i++)
      {
        int v1 = in(p + i_int2(circle_r3_h[i]) * scale).x;
        int v2 = in(p + i_int2(circle_r3_h[i+8]) * scale).x;

        int contrast = std::max(::abs(pv - v1), ::abs(pv - v2));
        // float contrast = (fabs(pv - v1) + fabs(pv - v2));
        if (max_contrast < contrast) max_contrast = contrast;

        int dev = ::abs(pv - (v1 + v2) / 2);
        // unsigned dev = ::abs(pv - v1) + ::abs(pv - v2);

	min_diff = std::min(min_diff, dev);
        mean_diff += dev;
      }

      if (max_contrast >= contrast_thresh)
      {
        min_diff = min_diff;
        mean_diff = 255*mean_diff / (8 * max_contrast);
      }
      else
      {
        min_diff = 0;
        mean_diff = 0;
      }

      return std::pair<int, int>(min_diff, max_contrast);
      // return mean_diff;
    }


    template <typename I>
    inline int compute_saliency2(i_short2 p, const I& in, int scale, float contrast_thresh)
    {
      int min_diff = 9999999;
      int mean_diff = 0.f;
      int pv = in(p).x;
      int max_contrast = 0;
      for(int i = 0; i < 8; i++)
      {
        const int& v1 = in(p + i_int2(circle_r3_h[i]) * scale).x;
        const int& v2 = in(p + i_int2(circle_r3_h[i+8]) * scale).x;

        // float contrast = std::max(fabs(pv - v1), fabs(pv - v2));
	int contrast = (::abs(pv - v1) + ::abs(pv - v2));
        if (max_contrast < contrast) max_contrast = contrast;

        int dev = ::abs(pv - (v1 + v2) / 2);
        // unsigned dev = ::abs(pv - v1) + ::abs(pv - v2);

	min_diff = std::min(min_diff, dev);
        mean_diff += dev;
      }

      if (max_contrast >= contrast_thresh)
      {
        min_diff = 255 * min_diff / max_contrast;
        mean_diff = 255 * mean_diff / (8 * max_contrast);
      }
      else
      {
        min_diff = 0;
        mean_diff = 0;
      }

      return min_diff;
      // return mean_diff;
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
      contrast_(d),
      new_points_(d),
      saliency_mode_(MAX),
      input_s2_(d),
      tmp_(d)
  {
  }


  mdfl_1s_detector::mdfl_1s_detector(const mdfl_1s_detector& d)
  {
    *this = d;
  }

  // mdfl_1s_detector&
  // mdfl_1s_detector::operator=(const mdfl_1s_detector& d)
  // {
  //   saliency_mode_ = d.saliency_mode_;
  //   contrast_th_ = d.contrast_th_;
  //   dev_th_ = d.dev_th_;
  //   saliency_ = clone(d.saliency_);
  //   new_points_ = clone(d.new_points_);
  // }

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

  mdfl_1s_detector&
  mdfl_1s_detector::set_saliency_mode(saliency_mode m)
  {
    saliency_mode_ = m;
    return *this;
  }

  void
  mdfl_1s_detector::update(const host_image2d<gl8u>& input)
  {
    START_PROF(mdfl_compute_saliency);

    mt_apply2d(sizeof(i_float1), input.domain() - border(8),
	       [this, &input] (i_int2 p)
	       {
		 std::pair<int, int> r = mdfl::compute_saliency(p, input, 1, contrast_th_);
		 contrast_(p) = r.second;
		 if (r.second > 0)
		   saliency_(p) = 255 * r.first / r.second;
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
                 if (contrast_(p) < contrast_th_) return;
                 if (saliency_(p) < dev_th_) return;

                 for (int i = 0; i < 8; i++)
                 {
                   i_int2 n(p + i_int2(c8[i]));
                   if (saliency_(p) < saliency_(n) || pset.has(n))
                     return;
                 }

                 new_points_(p) = 1;
               }, arch::cpu());

    st_apply2d(sizeof(char), saliency_.domain() - border(8),
               [this, &feature, &pset] (i_int2 p)
               {
                 if (new_points_(p)) pset.add(p, feature(p));
               }, arch::cpu());

  }


  mdfl_2s_detector::mdfl_2s_detector(const obox2d& d)
    : mdfl_1s_detector(d)
  {
  }

  void
  mdfl_2s_detector::update(const host_image2d<gl8u>& input)
  {
    START_PROF(mdfl_compute_saliency);

    dim3 dimblock = ::cuimg::dimblock(arch::cpu(), sizeof(i_uchar1), input.domain());
    local_jet_static_<0, 0, 1, 1>::run(input, input_s2_, tmp_, 0, dimblock);
    mt_apply2d(sizeof(i_float1), input.domain() - border(8),
	       [this, &input] (i_int2 p)
	       {
		 std::pair<int, int> r1 = mdfl::compute_saliency(p, input, 1, contrast_th_);
		 std::pair<int, int> r2 = mdfl::compute_saliency(p, input_s2_, 2, contrast_th_);
		 int c = std::min(r1.second, r2.second);
		 contrast_(p) = c;
		 if (c > 0)
		   saliency_(p) = 255 * std::min(r1.first, r2.first) / c;
	       }, arch::cpu());

    END_PROF(mdfl_compute_saliency);
  }

}

#endif

