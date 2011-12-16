#ifndef CUIMG_LARGE_MVT_DETECTOR_H_
# define  CUIMG_LARGE_MVT_DETECTOR_H_

# include <vector>

# include <cuimg/gpu/image2d.h>
# include <cuimg/improved_builtin.h>
# include <cuimg/gpu/tracking/fast38_feature.h>
# include <cuimg/gpu/tracking/naive_local_matcher.h>

namespace cuimg
{

  template <typename V>
  class large_mvt_detector
  {
  public:
    typedef naive_local_matcher<fast38_feature> search_t;
    typedef typename search_t::particle particle_t;
    typedef obox2d<point2d<int> > domain_t;

    large_mvt_detector(const domain_t& d);

    void update(const image2d<V>& f);

  private:
    void estimate();

    std::vector<image2d<i_float1> >* p2_;
    std::vector<image2d<i_float1> >* p1_;

    std::vector<image2d<i_float1> > pyramid1_;
    std::vector<image2d<i_float1> > diff_pyramid_;
    std::vector<image2d<i_float1> > pyramid2_;
    std::vector<image2d<i_float1> > pyramid_tmp1_;
    std::vector<image2d<i_float1> > pyramid_tmp2_;
    image2d<i_float4> display_;
    image2d<i_float1> gl_frame_;

    image2d<i_float4> particles_;
    image2d<i_float4> particles2_;

    fast38_feature feature_; // Feature maker
    naive_local_matcher<fast38_feature> sa_; // Tracking algorithm

    static const int PS = 5;
  };

}

# include <cuimg/gpu/tracking/large_mvt_detector.hpp>

#endif // !  CUIMG_LARGE_MVT_DETECTOR_H_
