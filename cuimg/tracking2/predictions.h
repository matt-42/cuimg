#ifndef CUIMG_TRACKING_PREDICTIONS_H_
# define CUIMG_TRACKING_PREDICTIONS_H_

# include <cuimg/improved_builtin.h>

namespace cuimg
{

  template <typename P>
  inline i_int2 motion_based_prediction(const P& p, const i_short2& prev_cam_motion, const i_short2& cam_motion)
  {
    assert(p.age > 0);
    if (p.age == 1)
      return p.pos;
    else
      return p.pos + p.speed - prev_cam_motion + cam_motion;
  }

}

#endif
