#include <cuimg/tracking2/opencv_tracker.h>

namespace cuimg
{

  opencv_tracker::opencv_tracker()
    : surf_(4),
      //bfmatcher_(cv::NORM_L2),
      bfmatcher_(cv::NORM_HAMMING)
  {
    nframe_ = 0;
    //detector_ = cv::FeatureDetector::create(method);
    extractor_ = cv::DescriptorExtractor::create("BRIEF");
    adapter_ = cv::AdjusterAdapter::create("FAST");
  }

  opencv_tracker::~opencv_tracker()
  {
  }

  void
  opencv_tracker::run(const host_image2d<gl8u>& in)
  {
    std::vector<cv::DMatch> dmatches;
    std::vector<std::vector<cv::DMatch> > ddmatches;
    cv::Mat in_cv(in);
    cv::Mat mask(in_cv.size(), 0, CV_8U);
    cv::Mat status, err;

    new_keypoints_.clear();

    //surf_(in_cv, mask, new_keypoints_, new_descriptors_);
    //orb_(in_cv, mask, new_keypoints_, new_descriptors_);
    //detector_->detect(in_cv, new_keypoints_);
    adapter_->detect(in_cv, new_keypoints_);
    int n = new_keypoints_.size();
    int np = 1000;
    if (n < np)
      adapter_->tooFew(np, n);
    else
      adapter_->tooMany(np, n);

    extractor_->compute(in_cv, new_keypoints_, new_descriptors_);

    // if (in_prev)
    // {
    //   calcOpticalFlowPyrLK(cv::Mat(in_prev), in_cv, keypoints_, new_keypoints_, status, err);
    // }
    if (keypoints_.size() > 0)
      bfmatcher_.match(new_descriptors_, descriptors_, dmatches);
      //matcher.match(new_descriptors_, descriptors_, dmatches);
      //matcher.radiusMatch(descriptors_, new_descriptors_, ddmatches, 10.f);

    matches_.resize(keypoints_.size());
    distances_.resize(keypoints_.size());
    std::fill(matches_.begin(), matches_.end(), -1);
    std::fill(distances_.begin(), distances_.end(), 99999.f);
    for (const auto& m : dmatches)
    {
      //std::cout << m.queryIdx << " " << m.trainIdx << " " << m.distance << std::endl;
      const cv::KeyPoint& src = keypoints_[m.trainIdx];
      const cv::KeyPoint& dst = new_keypoints_[m.queryIdx];
      if (m.distance < distances_[m.trainIdx] && norm(src.pt - dst.pt) < 30)
	{
	  matches_[m.trainIdx] = m.queryIdx;
	  distances_[m.trainIdx] = m.distance;
	}
    }

    for (const auto& dm : ddmatches)
    {
      for (const auto& m : dm)
	if (m.distance < distances_[m.trainIdx])
	{
	  matches_[m.trainIdx] = m.queryIdx;
	  distances_[m.trainIdx] = m.distance;
	}
    }

    descriptors_ = new_descriptors_;
    keypoints_.swap(new_keypoints_);

    in_prev = in;

    nframe_++;
  }

}
