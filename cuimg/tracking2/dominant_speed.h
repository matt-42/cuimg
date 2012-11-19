#ifndef CUIMG_DOMINANT_SPEED_H_
# define CUIMG_DOMINANT_SPEED_H_

# include <cuimg/improved_builtin.h>

namespace cuimg
{

  template <typename PI>
  inline __host__ __device__
  void dominant_speed(PI& pset)
  {
    SCOPE_PROF(dominant_speed);

    particle& part = pset[i];

    particle& buddy = pset(part.pos);
    if (buddy.age > part.age and norml2(part.speed - buddy.speed) < 2.f)
      part.age = 0;
  }

}

# include <cuimg/tracking2/gradient_descent_matcher.hpp>

#endif
