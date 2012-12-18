#ifndef CUIMG_DOMINANT_SPEED_H_
# define CUIMG_DOMINANT_SPEED_H_

# include <cuimg/improved_builtin.h>

namespace cuimg
{

  struct dominant_speed_estimator
  {
    inline dominant_speed_estimator(const obox2d& d);

    inline dominant_speed_estimator(const dominant_speed_estimator& d);
    inline dominant_speed_estimator& operator=(const dominant_speed_estimator& d);

    template <typename PI>
    inline __host__ __device__ i_short2 estimate(const PI& pset, i_int2 prev_camera_motion);

  private:
    host_image2d<unsigned short> h;
  };

}

# include <cuimg/tracking2/dominant_speed_estimator.hpp>

#endif
