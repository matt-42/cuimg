#ifndef CUIMG_TRACKING_PREDICTIONS_H_
# define CUIMG_TRACKING_PREDICTIONS_H_

# include <cuimg/improved_builtin.h>

namespace cuimg
{

  template <typename P>
  inline __host__ __device__
  i_int2 motion_based_prediction(const P& p, const i_short2& prev_cam_motion = i_short2(0,0),
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
  inline __host__ __device__
  i_int2 multiscale_prediction(const P& p, S& uf, int flow_ratio,
			       const i_short2& u_prev_cam_motion = i_short2(0,0),
			       const i_short2& u_cam_motion = i_short2(0,0))
  {
    if (uf.data())
      if (uf(p.pos / (2 * flow_ratio)) != NO_FLOW)
				return p.pos + 2 * uf(p.pos / (2 * flow_ratio));
      else
				return motion_based_prediction(p, u_prev_cam_motion*2, u_cam_motion*2);
		//return motion_based_prediction(p);
    else
      return motion_based_prediction(p);
  }

  // Need S::get_flow_at
  template <typename S, typename P>
  inline __host__ __device__
  i_int2 multiscale_prediction_old(const P& p, S& uf, int flow_ratio,
			       const i_short2& u_prev_cam_motion = i_short2(0,0),
			       const i_short2& u_cam_motion = i_short2(0,0))
  {

    if (p.age > 0)
      if (uf.data())
	return motion_based_prediction(p, u_prev_cam_motion*2, u_cam_motion*2);
      else
	return motion_based_prediction(p);
    else
      if (uf.data() && uf(p.pos / (2 * flow_ratio)) != NO_FLOW)
      {
    	//return motion_based_prediction(p, u_prev_cam_motion*2, u_cam_motion*2);
    	return p.pos + 2 * uf(p.pos / (2 * flow_ratio));
    	//return p.pos + 2 * uf(p.pos / (2 * flow_ratio));
      }
      else
    	return motion_based_prediction(p);
  }


}

#endif
