#ifndef CUIMG_REGULAR_DETECTOR_H_
# define CUIMG_REGULAR_DETECTOR_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  class regular_detector
  {
  public:
    inline regular_detector(const obox2d& d);
    inline regular_detector(const regular_detector& d);

    template <typename J>
    inline void update(const host_image2d<gl8u>& input, const J& mask);

    template <typename F, typename PS>
    inline void new_particles(const F& feature, PS& pset);

    inline regular_detector& set_step(int f);
    inline regular_detector& set_grad_th(int s);

    inline const host_image2d<int>& saliency() { return saliency_; }
    inline const host_image2d<gl8u>& contrast() { return contrast_; }

    inline int border_needed() const { return 3; }

  protected:
    float contrast_th_;
    host_image2d<int> saliency_;
    host_image2d<gl8u> contrast_;
    host_image2d<char> new_points_;
    host_image2d<gl8u> input_s2_;
    host_image2d<gl8u> tmp_;
    int step_;
    int grad_th_;
  };

}

# include <cuimg/tracking2/regular_detector.hpp>

#endif
