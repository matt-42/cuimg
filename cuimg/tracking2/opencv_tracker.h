
#ifndef OPENCV_TRACKER_H_

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <cuimg/cpu/host_image2d.h>
#include <cuimg/gl.h>

namespace cuimg
{

  class opencv_tracker
  {
  public:
    opencv_tracker();
    ~opencv_tracker();

    void run(const host_image2d<gl8u>& in);

    const std::vector<int>& matches() const            { return matches_; }
    const std::vector<cv::KeyPoint>& keypoints() const { return keypoints_; }

  private:
    std::vector<int> matches_;
    std::vector<float> distances_;
    cv::Ptr<cv::DescriptorExtractor> extractor_;
    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::AdjusterAdapter> adapter_;
    cv::SURF surf_;
    std::vector<cv::KeyPoint> keypoints_, new_keypoints_;
    cv::Mat descriptors_, new_descriptors_;
    cv::FlannBasedMatcher matcher;
    cv::BFMatcher bfmatcher_;

    host_image2d<gl8u> in_prev;

    unsigned nframe_;
  };

}

#endif
