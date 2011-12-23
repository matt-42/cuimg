#ifndef CUIMG_LARGE_MVT_DETECTOR_H_
# define  CUIMG_LARGE_MVT_DETECTOR_H_

# include <vector>

# include <cuimg/gpu/image2d.h>
# include <cuimg/cpu/host_image2d.h>
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

    large_mvt_detector();

    i_short2 estimate(const host_image2d<i_short2>& matches);

    void display();

  private:
    struct mvt
    {
      mvt(i_int2 p, i_char2 t) : pos(p), tr(t) {}

      i_int2 pos;
      i_char2 tr;
    };

    std::vector<mvt> mvts;
    host_image2d<unsigned short> h;
    int tr_max_cpt_;
    i_char2 tr_max_;
  };

}

# include <cuimg/gpu/tracking/large_mvt_detector.hpp>

#endif // !  CUIMG_LARGE_MVT_DETECTOR_H_
