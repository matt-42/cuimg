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
      strategy_(d)
  {
    lower_tracker_ = 0;
    upper_tracker_ = 0;
    if (nscales > 1)
    {
      upper_tracker_ = new tracker<F>(d / 2, this, nscales - 1);
      strategy_.set_upper(&upper_tracker_->strategy_);
    }

    strategy_.init();
  }

  template <typename F>
  tracker<F>::tracker(const obox2d& d, tracker<F>* lower, int nscales)
    : input_(d),
      pset_(d),
      strategy_(d)
  {
    lower_tracker_ = lower;
    upper_tracker_ = 0;
    if (nscales > 1)
    {
      upper_tracker_ = new tracker<F>(d / 2, this, nscales - 1);
      strategy_.set_upper(&upper_tracker_->strategy_);
    }

    strategy_.init();
  }

  template <typename F>
  tracker<F>::~tracker()
  {
    if (upper_tracker_)
      delete upper_tracker_;
  }

  template <typename F>
  tracker<F>& tracker<F>::update_input(const I& in)
  {
    copy(in, input_);
    if (upper_tracker_)
      upper_tracker_->subsample_input(in);
    return *this;
  }

  template <typename F>
  void tracker<F>::subsample_input(const I& in)
  {
    subsample(in, input_);
    if (upper_tracker_)
      upper_tracker_->subsample_input(input_);
  }

  template <typename F>
  tracker<F>& tracker<F>::run()
  {
    if (upper_tracker_)
      upper_tracker_->run();

    strategy_.update(input_, pset_);
    return *this;
  }


  template <typename F>
  void tracker<F>::clear()
  {
    if (upper_tracker_)
      upper_tracker_->clear();

    pset_.clear();
  }


}

#endif





