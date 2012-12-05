#ifndef CUIMG_MDFL_DETECTOR_H_
# define CUIMG_MDFL_DETECTOR_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  class mdfl_1s_detector
  {
  public:
    enum saliency_mode
    {
      MAX = 1,
      ADD = 2
    };

    inline mdfl_1s_detector(const obox2d& d);

    inline void update(const host_image2d<gl8u>& input);

    template <typename F, typename PS>
    inline void new_particles(const F& feature, PS& pset);

    inline mdfl_1s_detector& set_contrast_threshold(float f);
    inline mdfl_1s_detector& set_dev_threshold(float f);
    inline mdfl_1s_detector& set_saliency_mode(saliency_mode m);

    inline const host_image2d<gl01f>& saliency() { return saliency_; }

  private:
    saliency_mode saliency_mode_;
    float contrast_th_;
    float dev_th_;
    host_image2d<gl01f> saliency_;
    host_image2d<char> new_points_;
  };

}

# include <cuimg/tracking2/mdfl_detector.hpp>

#endif
