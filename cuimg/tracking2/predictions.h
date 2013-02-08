#ifndef CUIMG_TRACKING_PREDICTIONS_H_
# define CUIMG_TRACKING_PREDICTIONS_H_

# include <cuimg/improved_builtin.h>

namespace cuimg
{

  template <typename P>
    inline i_int2 motion_based_prediction(const P& p, const i_short2& prev_cam_motion = i_short2(0,0),
					  const i_short2& cam_motion = i_short2(0,0))
  {
    assert(p.age > 0);
    if (p.age == 1)
      return p.pos;
    else
      return p.pos + p.speed - prev_cam_motion + cam_motion;
  }

  // Need S::get_flow_at
  template <typename S, typename P>
  inline i_int2 multiscale_prediction(S& s, const P& p)
  {
    S* upper = static_cast<S*>(s.upper());
      if (p.age > 1)
	if (upper)
	  return motion_based_prediction(p, upper->prev_camera_motion()*2, upper->camera_motion()*2);
	else
	  return motion_based_prediction(p);
      else
	if (upper)
	{
	  return p.pos + 2 * upper->get_flow_at(p.pos / 2);
	}
	else
	  return p.pos;
  }

}

#endif

