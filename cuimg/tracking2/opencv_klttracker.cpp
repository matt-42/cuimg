#include <cuimg/neighb2d.h>
#include <cuimg/tracking2/opencv_klttracker.h>
#include <cuimg/cpu/fill.h>
#include <cuimg/cpu/copy.h>

namespace cuimg
{

  opencv_klttracker::opencv_klttracker(const obox2d& d)
    : fast_adapter_(10),
      pset_(d),
      mask_(d),
      detector_frequency_(1)
  {
    nframe_ = 0;
    adapter_ = cv::AdjusterAdapter::create("FAST");
  }

  opencv_klttracker::~opencv_klttracker()
  {
  }

  opencv_klttracker&
  opencv_klttracker::set_detector_frequency(unsigned nframe)
  {
    detector_frequency_ = nframe;
    return *this;
  }

  void
  opencv_klttracker::detect_keypoints(const host_image2d<gl8u>& in)
  {
    SCOPE_PROF(detect_keypoints)
    typedef unsigned char UC;
    cuimg::fill(mask_, UC(1));
    for (auto p : keypoints_)
    {
      i_int2 pos(p.y, p.x);
      for_all_in_static_neighb2d(pos, n, c8_h) if (mask_.has(n))
        mask_(n) = 0;
    }

    unsigned nkeypoints = 8000;
    std::vector<cv::KeyPoint> kps;
    //adapter_->detect(cv::Mat(in), kps, mask_);
    fast_adapter_.detect(cv::Mat(in), kps, mask_);
    for (auto& p : kps)
    {
      keypoints_.push_back(p.pt);
      pset_.add(i_float2(p.pt.y, p.pt.x), 0);
    }
    if (keypoints_.size() > nkeypoints)
      fast_adapter_.tooMany(nkeypoints * 1.1f, keypoints_.size());
    else
      if (keypoints_.size() < nkeypoints)
	fast_adapter_.tooFew(nkeypoints, keypoints_.size());
      else
	fast_adapter_.good();
    pset_.after_new_particles();
  }

  void
  opencv_klttracker::run(const host_image2d<gl8u>& in)
  {

    if (in_prev && keypoints_.size() > 0)
    {
      SCOPE_PROF(match_keypoints);


      std::vector<unsigned char> status;
      cv::Mat err;
      new_keypoints_.clear();
      keypoints_.clear();
      pset_type::kernel_type pset = pset_;
      for (unsigned i = 0; i < pset_.dense_particles().size(); i++)
      {
        i_float2 pos = pset.dense_particles()[i].pos;
        cv::Point2f pt(pos.y, pos.x);
        keypoints_.push_back(pt);
      }
      //std::cout << pset_.dense_particles().size() << std::endl;
      calcOpticalFlowPyrLK(cv::Mat(in_prev), cv::Mat(in), keypoints_, new_keypoints_, status, err, cv::Size(21,21), 3);
      matches_.resize(keypoints_.size());

      keypoints_.clear();
      pset_.before_matching();
      std::fill(matches_.begin(), matches_.end(), -1);
      for (unsigned i = 0; i < matches_.size(); i++)
      {
        //if (status.at<int>(0,i))
        if (status[i] && pset.dense_particles()[i].age > 0 && in.has(i_int2(new_keypoints_[i].y, new_keypoints_[i].x)) && err.at<float>(i) < 25)
        {
          //std::cout << status.at<int>(0,i) << " " << new_keypoints_[i] << std::endl;
	  // std::cout << int(err.at<float>(i)) << " " << keypoints_[i] << " " << new_keypoints_[i] << std::endl;
          pset.move(i, i_float2(new_keypoints_[i].y, new_keypoints_[i].x), 0);
          keypoints_.push_back(new_keypoints_[i]);
          matches_[i] = keypoints_.size() - 1;
        }
        else
          // std::cout << "NOT FOUND: "<< status.at<int>(0,i) << " " << new_keypoints_[i] << std::endl;
          //std::cout << "NOT FOUND: "<< int(status[i]) << " " << new_keypoints_[i] << std::endl;
          pset.remove(i);
      }
      // std::cout << new_keypoints_.size() << " " << new_keypoints_.size();
      pset_.after_matching();
    }

    //if (nframe_ == 0)

    //keypoints_.swap(new_keypoints_);

    if (!(nframe_ % detector_frequency_))
        detect_keypoints(in);
    if (!in_prev)
      in_prev = host_image2d<gl8u>(in.domain());
    copy(in, in_prev);

    nframe_++;
  }

}
