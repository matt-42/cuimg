#ifndef CUIMG_TRACKING_PREDICTIONS_H_
# define CUIMG_TRACKING_PREDICTIONS_H_

# include <cuimg/improved_builtin.h>

namespace cuimg
{

  template <typename P>
  inline __host__ __device__
  i_int2 motion_based_prediction(const P& p)
  {
    assert(p.age > 0);
    if (p.age == 1)
      return i_int2(0,0);
    else
      return p.speed;
  }

  // Need S::get_flow_at
  template <typename S, typename P>
  inline __host__ __device__
  i_int2 multiscale_prediction(const P& p, S& uf, int flow_ratio)
  {
    return motion_based_prediction(p);
    if (uf.data())
      if (uf(p.pos / (2 * flow_ratio)) != NO_FLOW)
	return uf(p.pos / (2 * flow_ratio)) * 2;
      else
	return motion_based_prediction(p);
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


  // Need S::get_flow_at
  template <typename S, typename P>
  inline __host__ __device__
  typename S::transform recursive_prediction(const P& p, S* upper)
  {
    if (upper)
      return upper->get_flow_at(p.pos / 2.).scale_transform(2);
    else
    {
      if (p.age == 1)
        return S::transform::identity();
      else
        return p.speed;
    }
  }
}

#endif
