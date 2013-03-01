#ifndef CUIMG_FAST_DETECTOR_H_
# define CUIMG_FAST_DETECTOR_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  class fast_detector
  {
  public:
    inline fast_detector(const obox2d& d);

    inline void update(const host_image2d<gl8u>& input);

    template <typename F, typename PS>
    inline void new_particles(const F& feature, PS& pset);

    inline fast_detector& set_fast_threshold(float f);
    inline fast_detector& set_n(unsigned n);

    inline const host_image2d<gl8u>& saliency() { return saliency_; }
    inline const host_image2d<gl8u>& contrast() { return contrast_; }

  private:
    float contrast_th_;
    float fast_th_;
    unsigned n_;
    host_image2d<gl8u> saliency_;
    host_image2d<gl8u> contrast_;
    host_image2d<gl8u> input_;
    host_image2d<char> new_points_;
  };

}

# include <cuimg/tracking2/fast_detector.hpp>

#endif
