#ifndef CUIMG_LARGE_MVT_DETECTOR_H_
# define  CUIMG_LARGE_MVT_DETECTOR_H_

# include <vector>

# include <thrust/host_vector.h>

# include <cuimg/gpu/device_image2d.h>
# include <cuimg/cpu/host_image2d.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{

  template <typename V>
  class large_mvt_detector
  {
  public:

    typedef obox2d<point2d<int> > domain_t;

    large_mvt_detector();

    template <typename P>
    i_short2 estimate(const thrust::host_vector<P>& particles,
                      unsigned n_particles);

    void display();

  private:

    host_image2d<unsigned short> h;
    int tr_max_cpt_;
    i_char2 tr_max_;
    unsigned n_particles_;
  };

}

# include <cuimg/tracking/large_mvt_detector.hpp>

#endif // !  CUIMG_LARGE_MVT_DETECTOR_H_
