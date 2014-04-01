#ifndef CUIMG_RIGID_TRANSFORM_HPP_
# define CUIMG_RIGID_TRANSFORM_HPP_

# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/fill.h>
# include <opencv2/opencv.hpp>
# include <cuimg/tracking2/particle_container.h>

namespace cuimg
{

  rigid_transform_estimator::rigid_transform_estimator()
  {
  }

  rigid_transform_estimator::rigid_transform_estimator(const rigid_transform_estimator& d)
  {
  }

  // rigid_transform_estimator&
  // rigid_transform_estimator::operator=(const rigid_transform_estimator& d)
  // {
  //   h = host_image2d<unsigned short>(d.h.domain());
  //   copy(d.h, h);
  //   return *this;
  // }

  template <typename T>
  inline cv::Mat_<float> to_cv_mat(const improved_builtin<T, 2>& v)
  {
    return cv::Mat_<float>(3, 1) << v.x, v.y, 1;
  }

  template <typename T>
  inline cv::Mat_<float> to_cv_mat2(const improved_builtin<T, 2>& v)
  {
    return cv::Mat_<float>(2, 1) << v.x, v.y;
  }

  inline cv::Mat_<float> to_cv_mat2(const cv::Mat_<float>& v)
  {
    return cv::Mat_<float>(2, 1) << v(0, 0), v(1, 0);
  }

  template <typename PI>
  cv::Mat
  rigid_transform_estimator::estimate(const PI& pset, const cv::Mat& prev_camera_motion)
  {
    SCOPE_PROF(rigid_transform_estimation);

    std::vector<cv::Point2f> src, dst;
    // cv::Mat_<int> src(pset.size(), 2), dst(pset.size(), 2);
    // cv::Mat_<unsigned char> src(0, 2), dst(0, 2);
    unsigned npart = 0;
    for (unsigned i = 0; i < pset.dense_particles().size(); i++)
    {
      const particle& part = pset.dense_particles()[i];
      if (part.age > 10)
      {
	cv::Mat_<float> d = to_cv_mat2(part.pos + part.speed);
	src.push_back(cv::Point2f(part.pos.r(), part.pos.c()));
	dst.push_back(cv::Point2f(d(0,0), d(1,0)));
	npart++;
      }
    }

    if (npart > 10) 
    {
      return estimateRigidTransform(src, dst, true);
    }
    else
      return cv::Mat_<float>(2, 3) << 1.f,0.f,0.f, 0,1.f,0.f;
  }

}

#endif


