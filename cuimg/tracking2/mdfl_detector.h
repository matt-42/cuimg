#ifndef CUIMG_MDFL_DETECTOR_H_
# define CUIMG_MDFL_DETECTOR_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  class mdfl_1s_detector
  {
  public:
    inline mdfl_1s_detector(const obox2d& d);

    inline void update(const host_image2d<i_uchar1>& input);

    template <typename F, typename PS>
    inline void new_particles(const F& feature, PS& pset, float contrast_th, float dev_th);

  private:
    host_image2d<float> saliency_;
  };

}

# include <cuimg/tracking2/mdfl_detector.hpp>

#endif
