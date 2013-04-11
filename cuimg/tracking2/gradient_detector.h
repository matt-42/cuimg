#ifndef CUIMG_GRADIENT_DETECTOR_H_
# define CUIMG_GRADIENT_DETECTOR_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  class gradient_detector
  {
  public:
    inline gradient_detector(const obox2d& d);
    inline gradient_detector(const gradient_detector& d);

    template <typename J>
    inline void update(const host_image2d<gl8u>& input, const J& mask);

    template <typename F, typename PS>
    inline void new_particles(const F& feature, PS& pset);

    inline gradient_detector& set_grad_th(int s);
    inline gradient_detector& set_sigma(float s);

    inline const host_image2d<int>& saliency() { return saliency_; }
    inline const host_image2d<gl8u>& contrast() { return contrast_; }

    inline int border_needed() const { return 3; }

  protected:
    float contrast_th_;
    host_image2d<int> saliency_;
    host_image2d<gl8u> contrast_;
    host_image2d<gl8u> input_s2_;
    host_image2d<char> new_points_;
    host_image2d<gl8u> tmp_;
    int grad_th_;
    float sigma_;
  };

}

# include <cuimg/tracking2/gradient_detector.hpp>

#endif
