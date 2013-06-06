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
    // static int with_upper_flow = 0;
    // static int without_upper_flow = 0;

    // if (!((with_upper_flow+without_upper_flow) % 1000))
    // 	std::cout << with_upper_flow << " " << without_upper_flow << " " << 100.f*without_upper_flow / with_upper_flow << std::endl;

    if (uf.data())
      if (uf(p.pos / (2 * flow_ratio)) != NO_FLOW)
      {
	//with_upper_flow++;
	//return motion_based_prediction(p, u_prev_cam_motion*2, u_cam_motion*2);
	return p.pos + 2 * uf(p.pos / (2 * flow_ratio));
      }
      else
      {
	//return motion_based_prediction(p, u_prev_cam_motion*2, u_cam_motion*2);
	// look in c8
	// for_all_in_static_neighb2d(p.pos / (2 * flow_ratio), n, c8_h)
	//   if (uf.has(n) && uf(n) != NO_FLOW)
	//   {
	//     //with_upper_flow++;
	//     return p.pos + 2 * uf(n);
	//   }

	// without_upper_flow++;
	return motion_based_prediction(p);
	//return motion_based_prediction(p, u_prev_cam_motion*2, u_cam_motion*2);
      }
    //return motion_based_prediction(p, u_prev_cam_motion*2, u_cam_motion*2);
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
