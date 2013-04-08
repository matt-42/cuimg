#ifndef CUIMG_DENSE_DETECTOR_H_
# define CUIMG_DENSE_DETECTOR_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  class dense_detector
  {
  public:
    inline dense_detector(const obox2d& d);

    template <typename J>
    inline void update(const host_image2d<gl8u>& input, const J& mask);

    template <typename F, typename PS>
    inline void new_particles(const F& feature, PS& pset);

    inline int border_needed() const { return 0; }

  };

}

# include <cuimg/tracking2/dense_detector.hpp>

#endif
