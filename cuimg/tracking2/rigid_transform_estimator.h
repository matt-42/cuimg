#ifndef CUIMG_RIGID_TRANSFORM_H_
# define CUIMG_RIGID_TRANSFORM_H_

# include <cuimg/improved_builtin.h>
# include <opencv2/core/core.hpp>

namespace cuimg
{

  struct rigid_transform_estimator
  {
    inline rigid_transform_estimator();
    inline rigid_transform_estimator(const rigid_transform_estimator& d);
    inline rigid_transform_estimator& operator=(const rigid_transform_estimator& d);

    template <typename PI>
    inline __host__ __device__ cv::Mat estimate(const PI& pset, const cv::Mat& prev_camera_motion);

  private:
  };

}

# include <cuimg/tracking2/rigid_transform_estimator.hpp>

#endif
