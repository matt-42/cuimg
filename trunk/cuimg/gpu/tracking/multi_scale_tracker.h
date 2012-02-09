#ifndef CUIMG_MULTI_SCALE_TRACKER_H_
# define CUIMG_MULTI_SCALE_TRACKER_H_

# include <cuimg/cpu/host_image2d.h>
# include <cuimg/gpu/image2d.h>

# include <cuimg/gpu/tracking/large_mvt_detector.h>
# include <cuimg/gpu/tracking/trajectory_tracer.h>

# include <cuimg/gpu/tracking/global_mvt_thread.h>

namespace cuimg
{

  template <typename F, template <class> class SA_>
  class multi_scale_tracker
  {
  public:
    typedef obox2d<point2d<int> > D;
    //typedef fast382s_feature F;
    //typedef fast16_2s_feature F;
    //typedef lgp82s_feature F;
    //typedef naive_local_matcher<F> SA;
    //typedef naive_local_matcher2<F> SA;
    typedef SA_<F> SA;
    typedef typename SA::particle P;

    multi_scale_tracker(const D& d);
    ~multi_scale_tracker();

    const D& domain() const;

    void update(const host_image2d<i_uchar3>& in);

    const image2d<P>& particles() const
    {
      return matcher_[0]->particles();
    }

    const image2d<char>& errors() const
    {
      return matcher_[0]->errors();
    }


  private:
    image2d<i_uchar3> frame_uc3_;
    image2d<i_uchar1> frame_;
    std::vector<image2d<i_uchar1> > pyramid_;
    std::vector<image2d<i_uchar1> > pyramid_tmp1_;
    std::vector<image2d<i_uchar1> > pyramid_tmp2_;

    std::vector<F*> feature_;
    std::vector<SA*> matcher_;

    static const unsigned PS = 4;

    std::vector<image2d<i_uchar4> > pyramid_display1_;
    std::vector<image2d<i_uchar4> > pyramid_display2_;
    std::vector<image2d<i_uchar4> > pyramid_speed_;

    image2d<i_short2> dummy_matches_;

    trajectory_tracer traj_tracer_;

    global_mvt_thread<SA> mvt_detector_thread_;
  };

}

# include <cuimg/gpu/tracking/multi_scale_tracker.hpp>

#endif // !CUIMG_MULTI_SCALE_TRACKER_H_
