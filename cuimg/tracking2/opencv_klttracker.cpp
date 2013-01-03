#include <cuimg/neighb2d.h>
#include <cuimg/tracking2/opencv_klttracker.h>
#include <cuimg/cpu/fill.h>
#include <cuimg/cpu/copy.h>

namespace cuimg
{

  opencv_klttracker::opencv_klttracker()
  {
    nframe_ = 0;
    adapter_ = cv::AdjusterAdapter::create("FAST");
  }

  opencv_klttracker::~opencv_klttracker()
  {
  }

  void
  opencv_klttracker::detect_keypoints(const host_image2d<gl8u>& in)
  {
    host_image2d<unsigned char> mask(in.domain());
    typedef unsigned char UC;
    cuimg::fill(mask, UC(1));
    for (auto p : keypoints_)
    {
      i_int2 pos(p.y, p.x);
      for_all_in_static_neighb2d(pos, n, c25) if (mask.has(n))
	mask(n) = 0;
    }

    std::vector<cv::KeyPoint> kps;
    adapter_->detect(cv::Mat(in), kps, mask);
    for (auto& p : kps)
      keypoints_.push_back(p.pt);
  }

  void
  opencv_klttracker::run(const host_image2d<gl8u>& in)
  {

    if (in_prev && keypoints_.size() > 0)
    {
      cv::Mat status, err;
      new_keypoints_.clear();
      calcOpticalFlowPyrLK(cv::Mat(in_prev), cv::Mat(in), keypoints_, new_keypoints_, status, err);
      matches_.resize(keypoints_.size());

      keypoints_.clear();
      std::fill(matches_.begin(), matches_.end(), -1);
      for (unsigned i = 0; i < matches_.size(); i++)
      {
	if (status.at<int>(0,i))
	{
	  keypoints_.push_back(new_keypoints_[i]);
	  matches_[i] = keypoints_.size() - 1;
	}
      }
      // std::cout << new_keypoints_.size() << " " << new_keypoints_.size();
    }

    //if (nframe_ == 0)

    //keypoints_.swap(new_keypoints_);

    detect_keypoints(in);
    if (!in_prev)
      in_prev = host_image2d<gl8u>(in.domain());
    copy(in, in_prev);

    nframe_++;
  }

}





