
#ifndef OPENCV_KLTTRACKER_H_

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <cuimg/cpu/host_image2d.h>
#include <cuimg/gl.h>

namespace cuimg
{

  class opencv_klttracker
  {
  public:
    opencv_klttracker();
    ~opencv_klttracker();

    void run(const host_image2d<gl8u>& in);
    void detect_keypoints(const host_image2d<gl8u>& in);

    const std::vector<int>& matches() const            { return matches_; }
    const std::vector<cv::Point2f>& keypoints() const { return keypoints_; }

  private:
    std::vector<int> matches_;
    std::vector<float> distances_;
    cv::Ptr<cv::AdjusterAdapter> adapter_;
    std::vector<cv::Point2f > keypoints_, new_keypoints_;

    host_image2d<gl8u> in_prev;

    unsigned nframe_;
  };

}

#endif
