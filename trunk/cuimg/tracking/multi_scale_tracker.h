#ifndef CUIMG_MULTI_SCALE_TRACKER_H_
# define CUIMG_MULTI_SCALE_TRACKER_H_

# include <cuimg/cpu/host_image2d.h>
# include <cuimg/gpu/device_image2d.h>

# include <cuimg/tracking/large_mvt_detector.h>
# include <cuimg/tracking/trajectory_tracer.h>

# include <cuimg/tracking/global_mvt_thread.h>

namespace cuimg
{

  template <typename F, typename SA>
  class multi_scale_tracker
  {
  public:
    typedef obox2d<point2d<int> > D;

    enum { target = F::target };

    typedef typename F::vtype V;
    typedef image2d_target(target, i_uchar3) image2d_uc3;
    typedef image2d_target(target, i_short2) image2d_s2;
    typedef image2d_target(target, i_float1) image2d_f1;
    typedef image2d_target(target, V) image2d_V;
    typedef image2d_target(target, i_float4) image2d_f4;
    typedef image2d_target(target, char) image2d_c;

    typedef typename SA::particle P;
    typedef typename SA::particle particle_type;

    typedef image2d_target(target, P) image2d_P;
    typedef typename SA::particle_vector particle_vector;
    typedef typename SA::i_short2_vector i_short2_vector;

    multi_scale_tracker(const D& d);
    ~multi_scale_tracker();

    const D& domain() const;

    void                   update(const host_image2d<i_uchar3>& in);
    const particle_vector& particles(unsigned scale) const;
    const image2d_P&       particles_img(unsigned scale) const;
    unsigned               nparticles(unsigned scale) const;
    const image2d_c&       errors() const;

  private:
    void display();

    image2d_uc3 frame_uc3_;
    image2d_V frame_;
    std::vector<image2d_V > pyramid_;
    std::vector<image2d_V > pyramid_tmp1_;
    std::vector<image2d_V > pyramid_tmp2_;

    std::vector<F*> feature_;
    std::vector<SA*> matcher_;

    static const unsigned PS = 4;

    std::vector<image2d_f4 > pyramid_display1_;
    std::vector<image2d_f4 > pyramid_display2_;
    std::vector<image2d_f4 > pyramid_speed_;

    image2d_s2 dummy_matches_;

    /* trajectory_tracer<target> traj_tracer_; */

    global_mvt_thread<SA> mvt_detector_thread_;
  };

}

# include <cuimg/tracking/multi_scale_tracker.hpp>

#endif // !CUIMG_MULTI_SCALE_TRACKER_H_
