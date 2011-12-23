#ifndef CUIMG_MULTI_SCALE_TRACKER_H_
# define CUIMG_MULTI_SCALE_TRACKER_H_

# include <cuimg/cpu/host_image2d.h>
# include <cuimg/gpu/image2d.h>

# include <cuimg/gpu/tracking/naive_local_matcher.h>
# include <cuimg/gpu/tracking/large_mvt_detector.h>
# include <cuimg/gpu/tracking/fast38_feature.h>
# include <cuimg/gpu/tracking/trajectory_tracer.h>

namespace cuimg
{

  class multi_scale_tracker
  {
  public:
    typedef obox2d<point2d<int> > D;
    typedef fast38_feature F;
    typedef naive_local_matcher<fast38_feature> SA;
    typedef typename SA::particle P;

    multi_scale_tracker(const D& d);
    ~multi_scale_tracker();

    const D& domain() const;

    void update(const host_image2d<i_uchar3>& in);

  private:
    image2d<i_uchar3> frame_uc3_;
    image2d<i_float1> frame_;
    std::vector<image2d<i_float1> > pyramid_;
    std::vector<image2d<i_float1> > pyramid_tmp1_;
    std::vector<image2d<i_float1> > pyramid_tmp2_;

    std::vector<host_image2d<i_short2> > matches_;

    std::vector<F*> feature_;
    std::vector<SA*> matcher_;

    static const unsigned PS = 5;

    large_mvt_detector<i_float1> mvt_detector;

    image2d<i_float4> p_display1_;
    image2d<i_float4> p_display2_;

    trajectory_tracer traj_tracer_;
  };

}

# include <cuimg/gpu/tracking/multi_scale_tracker.hpp>

#endif // !CUIMG_MULTI_SCALE_TRACKER_H_
