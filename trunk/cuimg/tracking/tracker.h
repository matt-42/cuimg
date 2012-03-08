#ifndef CUIMG_TRACKER_H_
# define CUIMG_TRACKER_H_


# include <cuimg/gpu/device_image2d.h>

namespace cuimg
{

  template <typename SA, typename F>
  class tracker
  {
  public:

    typedef typename F::feature_t feature_t;
    typedef typename SA::particle particle_t;
    tracker(SA& sa, F& f);

    template <typename V>
    void update(const device_image2d<V>& in);

  private:
    F& f_; // Feature maker
    SA& sa_; // Tracking algorithm

  };

}

# include <cuimg/tracking/tracker.hpp>

#endif // !CUIMG_TRACKER_H_
