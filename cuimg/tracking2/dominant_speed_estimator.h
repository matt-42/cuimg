#ifndef CUIMG_DOMINANT_SPEED_H_
# define CUIMG_DOMINANT_SPEED_H_

# include <cuimg/improved_builtin.h>

namespace cuimg
{

  template <typename A>
  struct dominant_speed_estimator
  {
    inline dominant_speed_estimator(const obox2d& d);

    inline dominant_speed_estimator(const dominant_speed_estimator& d);
    inline dominant_speed_estimator& operator=(const dominant_speed_estimator& d);

    template <typename PI>
    inline i_short2 estimate(const PI& pset, i_int2 prev_camera_motion, const cpu&);

#ifndef NO_CUDA
    template <typename PI>
    inline i_short2 estimate(const PI& pset, i_int2 prev_camera_motion, const cuda_gpu&);
#endif

  private:
    typename A::template image2d<int>::ret h;
    typename A::template vector<int>::ret vote_buffer_;
    typename A::template vector<int>::ret histo_values_;
    typename A::template vector<int>::ret histo_counts_;
  };

}

# include <cuimg/tracking2/dominant_speed_estimator.hpp>

#endif
