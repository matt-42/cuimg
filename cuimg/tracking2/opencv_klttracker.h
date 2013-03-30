
#ifndef OPENCV_KLTTRACKER_H_

#include <opencv2/opencv.hpp>
#include <cuimg/cpu/host_image2d.h>
#include <cuimg/gl.h>

#include <cuimg/tracking2/particle_container.h>


namespace cuimg
{

  namespace internal
  {
    struct dummy_feature
    {
    public:
      typedef int value_type;
      inline dummy_feature(int) {}
    };
  }

  class opencv_klttracker
  {
  public:
    typedef particle_container<internal::dummy_feature, particle_f> pset_type;
    opencv_klttracker(const obox2d& d);
    ~opencv_klttracker();

    opencv_klttracker& set_detector_frequency(unsigned nframe);

    void run(const host_image2d<gl8u>& in);
    void detect_keypoints(const host_image2d<gl8u>& in);

    const std::vector<int>& matches() const            { return matches_; }
    const std::vector<cv::Point2f>& keypoints() const { return keypoints_; }

    inline const obox2d& domain() const     { return pset_.domain(); }
    inline       pset_type& pset()     { return pset_; }
    inline const pset_type& pset() const { return pset_; }

    inline void clear() { pset_.clear(); }

  private:
    std::vector<int> matches_;
    std::vector<float> distances_;
    cv::FastAdjuster fast_adapter_;
    cv::Ptr<cv::AdjusterAdapter> adapter_;
    std::vector<cv::Point2f > keypoints_, new_keypoints_;

    host_image2d<gl8u> in_prev;

    pset_type pset_;
    host_image2d<unsigned char> mask_;

    unsigned nframe_;
    int detector_frequency_;
  };

}

#endif
