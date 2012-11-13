#ifndef CUIMG_TRACKER2_HPP_
# define CUIMG_TRACKER2_HPP_

# include <cuimg/copy.h>
# include <cuimg/gpu/mipmap.h>

namespace cuimg
{


  template <typename F>
  tracker<F>::tracker(const obox2d& d, int nscales)
    : input_(d),
      pset_(d),
      feature_(d),
      detector_(d),
      camera_motion_(0,0),
      prev_camera_motion_(0,0)
  {
    lower_tracker_ = 0;
    upper_tracker_ = 0;
    if (nscales > 1)
      upper_tracker_ = new tracker<F>(d / 2, nscales - 1);
  }

  template <typename F>
  tracker<F>::tracker(const obox2d& d, tracker<F>* lower, int nscales)
    : input_(d),
      feature_(d),
      detector_(d),
      camera_motion_(0,0),
      prev_camera_motion_(0,0)
  {
    lower_tracker_ = lower;
    upper_tracker_ = 0;
    if (nscales > 1)
      upper_tracker_ = new tracker<F>(d / 2, nscales - 1);
  }

  template <typename F>
  tracker<F>::~tracker()
  {
    if (upper_tracker_)
      delete upper_tracker_;
  }

  template <typename F>
  void tracker<F>::update_input(const I& in)
  {
    copy(in, input_);
    if (upper_tracker_)
      upper_tracker_->subsample_input(in);
  }

  template <typename F>
  void tracker<F>::subsample_input(const I& in)
  {
    subsample(in, input_);
    if (upper_tracker_)
      upper_tracker_->subsample_input(input_);
  }

  template <typename F>
  void tracker<F>::run()
  {
    feature_.update(input_);
    F::match_particles(*this);

    detector_.update(input_);
    F::new_particles(*this);

    if (lower_tracker_)
    {
      camera_motion_ = F::estimate_camera_motion(*this);
      lower_tracker_->run();
    }
  }


}

#endif
