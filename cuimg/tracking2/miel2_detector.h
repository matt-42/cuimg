#ifndef CUIMG_MIEL2_DETECTOR_H_
# define CUIMG_MIEL2_DETECTOR_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  class miel2_1s_detector
  {
  public:
    enum saliency_mode
    {
      MAX = 1,
      ADD = 2
    };

    inline miel2_1s_detector(const obox2d& d);
    inline miel2_1s_detector(const miel2_1s_detector& d);

    template <typename J>
    inline void update(const host_image2d<gl8u>& input, const J& mask);

    template <typename F, typename PS>
    inline void new_particles(const F& feature, PS& pset);

    inline miel2_1s_detector& set_contrast_threshold(float f);
    inline miel2_1s_detector& set_dev_threshold(float f);
    inline miel2_1s_detector& set_saliency_mode(saliency_mode m);

    inline float dev_threshold() const { return dev_th_; }

    inline const host_image2d<int>& saliency() { return saliency_; }
    inline const host_image2d<gl8u>& contrast() { return contrast_; }

    inline int border_needed() const { return 3; }

  protected:
    saliency_mode saliency_mode_;
    float contrast_th_;
    float dev_th_;
    host_image2d<int> saliency_;
    host_image2d<gl8u> contrast_;
    host_image2d<char> new_points_;
    host_image2d<gl8u> input_s2_;
    host_image2d<gl8u> tmp_;
  };

}

# include <cuimg/tracking2/miel2_detector.hpp>

#endif
